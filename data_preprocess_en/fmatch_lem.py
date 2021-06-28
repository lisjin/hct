#!/usr/bin/env python3
import argparse
import difflib
import json
import numpy as np
import regex

from collections import deque
from functools import partial
from itertools import chain
from nltk import Tree
from os import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool
from sacrebleu import sentence_chrf

from utils import eprint, fromstring, read_lem, read_expand, read_stop_phrs, tightest_span, Metrics, concat_path


def difflib_match(ctx, phr):
    """Fuzzy matching using https://docs.python.org/3/library/difflib.html#difflib.SequenceMatcher.get_matching_blocks"""
    ms = difflib.SequenceMatcher(None, ctx, phr).get_matching_blocks()
    a, _, n = max(ms, key=lambda m: m.size)
    return a, n


def regex_match(ctx, phr):
    """Fuzzy matching using https://pypi.org/project/regex/"""
    ret = None
    a, n = -1, 0
    try:
        # (?b) is BESTMATCH, (?e) is ENHANCEMATCH flag
        # NOTE: n.fuzzy_counts is (sub, ins, del) count
        m = regex.search(f'(?e)(?<!\\S){phr}(?!\\S){{e}}', ctx)
        if m:
            a, n = m.span()
            n -= a
    except regex._regex_core.error:  # no match
        pass
    return a, n


def fill_word_bnd(ctx):
    ctx_spl = ctx.split()
    cind = -1
    word_bnd = {cind: 0}
    for i, w in enumerate(ctx_spl):  # map char to token indices of spaces
        cind += len(w) + 1
        word_bnd[cind] = len(word_bnd)
    word_bnd[len(ctx)] = len(ctx_spl)
    return ctx_spl, word_bnd


def fmatch_single(phr_spl, ctx, tgt, cpt_tgt, phr_tgt_sp, match_fn, tmode, stop_phrs):
    ctx_spl, word_bnd = fill_word_bnd(ctx)
    offset, pl = phr_tgt_sp
    tgt_spl = tgt.split()
    for sp in stop_phrs:
        if phr_spl[:len(sp)] == sp and ' '.join(sp) not in ctx:
            phr_spl[:] = phr_spl[len(sp):]
            break
    phr = ' '.join(phr_spl)
    bspans = {}

    def check_span(a, n):
        if a > -1 and a < len(ctx):
            sphr = ctx[a:a+n]
            ek = a + n - 1
            while a > -1 and not ctx[a].isspace():  # closest space index <= a
                a -= 1
            while ek < len(ctx) and not ctx[ek].isspace():  # same >= a + n
                ek += 1
            sp = (word_bnd[a], word_bnd[ek])
            if ek - a < 2 * len(sphr):  # ignore bad matches
                return n, sp
        return 0, (-1, -1)

    def search_span(i, k):
        kp = k if i > -1 else k + i
        a, n = -1, 0
        if kp > 0:
            ip = max(0, i)
            cur_phr = ' '.join(phr_spl[ip:ip+kp])
            a, n = match_fn(ctx, cur_phr)
        n, sp = check_span(a, n)
        return a, n, sp

    def trav(t, i):
        """Compare tree span matches bottom-up, replacing parent score by
        children's as necessary."""
        a = -1  # start character index in context match
        n = 0  # number of matched characters in context
        k = len(t.leaves())  # span token length in phrase
        a, n, sp = search_span(i, k)
        bspans[(i, k)] = [(i, k, a, n, sp)]
        cn = 0
        spans = []
        i2 = i
        for st in t:
            if type(st) is Tree:
                k2 = len(st.leaves())
                if -k2 < i2 and i2 < pl:
                    n2 = trav(st, i2)
                    if n2 > 0:
                        spans.extend(bspans[(i2, k2)])
                        cn += n2
                i2 += k2
            else:
                i2 += 1
            if i2 >= pl:
                break
        if cn > n:
            bspans[(i, k)][:] = spans[:]
            n = cn
        return n

    def bottom_up():
        i, k = -offset, len(cpt_tgt.leaves())
        t, bsp = cpt_tgt, (i, k)
        tup = tightest_span(cpt_tgt, i, k, pl)
        if tup is not None:
            t, i, k = tup
        bsp = (i, k)
        _ = trav(t, i)
        return bsp

    def iter_children(t, i, k):
        spans, sts = [], []
        i2 = i
        cn = 0
        for st in t:
            if type(st) is Tree:
                k2 = len(st.leaves())
                cur_sp = (i2, k2)
                if -k2 < i2 and i2 < pl:
                    a2, n2, sp = search_span(*cur_sp)
                    bspans[cur_sp] = [(*cur_sp, a2, n2, sp)]
                    spans.append(bspans[cur_sp][-1])
                    sts.append((st, *cur_sp))
                    cn += n2
                i2 += k2
            else:
                i2 += 1
            if i2 >= pl:
                break
        return spans, sts, cn

    def expand_bspans(k, out=[]):
        if k in bspans:
            if len(bspans[k]) > 1:
                for sp in bspans[k]:
                    expand_bspans(sp[:2], out)
            elif len(bspans[k]):
                out.append(bspans[k][0])
        return out

    def top_down():
        """Stop exploring child spans if their summed match count is less than
        parent's. This helps minimize number of tree splits.
        """
        i, k = -offset, len(cpt_tgt.leaves())
        bsp = None
        if offset > -1:
            t, i, k = tightest_span(cpt_tgt, -offset, len(cpt_tgt.leaves()), pl)
            bsp = (i, k)
            bspans[bsp] = [(i, k, *search_span(i, k))]
            dq = deque([(t, i, k)])
            while dq:
                t, i, k = dq.pop()
                spans, sts, cn = iter_children(t, i, k)

                n = bspans[(i, k)][0][-2]
                if cn > n or n == 0:
                    bspans[(i, k)][:] = spans[:]
                    dq.extendleft(sts)
        bspans[bsp][:] = expand_bspans(bsp)
        return bsp

    bsp = bottom_up() if tmode == 'bup' else top_down()
    m, sps, n_sp = '', None, 0
    if bsp:
        bspans[bsp] = [t for t in bspans[bsp] if t[-2] > 0]
        sps = [t[-1] for t in bspans[bsp]]
        ctx_spl = ctx.split()
        m = list(chain.from_iterable([ctx_spl[t[0]:t[1]] for t in sps]))
        n_sp = len(bspans[bsp])
        if args.print_found:
            eprint(f'{n_sp}\t{" ".join(m)}\t{phr}')
    return m, phr, sps, n_sp


def fmatch(args):
    phrs, ctxs, tgts, cpts_tgt, phr_tgt_sps, stop_phrs = read_fs(args)

    match_fn = regex_match if args.mmode == 'regex' else difflib_match
    fms = partial(fmatch_single, match_fn=match_fn, tmode=args.tmode, stop_phrs=stop_phrs)
    if not args.debug:
        with Pool(args.n_proc) as p:
            res = p.map(fms, phrs, ctxs, tgts, cpts_tgt, phr_tgt_sps)
    else:
        res = [fms(*t) for t in zip(phrs, ctxs, tgts, cpts_tgt, phr_tgt_sps)]

    with open(concat_path(args, args.out_f.format(args.mmode, args.tmode)), 'w', encoding='utf8') as f:
        json.dump([t[2] for t in res], f)
    cands, _, _, n_sps = zip(*res)
    n_sps = np.array(n_sps)
    bleu_tup = Metrics.bleu_score(phrs, cands)
    chrfs = np.array([sentence_chrf(' '.join(m), [p]).score if m else 0. for m, p, _, _ in res])
    return (chrfs.mean(), chrfs.std()), (n_sps.mean(), n_sps.std())


def read_fs(args):
    """Returns (phrs, ctxs, tgts, phr tree IDs, unique phr trees, tgt trees)."""
    with Pool(args.n_proc) as p:
        with open(concat_path(args, 'cpts_tgt.txt'), encoding='utf8') as f:
            cpts_tgt = p.map(fromstring, [l.rstrip() for l in f])
    ctxs, tgts = read_lem(concat_path(args, 'unmatch_lems.json'))
    cpts_tgt, ctxs, tgts, phrs, phr_tgt_sps = read_expand(concat_path(args, 'phr_tgt_sps.json'),
            ctxs, tgts, cpts_tgt)
    phr_tgt_sps = [sp for pts in phr_tgt_sps for sp in pts]
    stop_phrs = read_stop_phrs(args.stop_phrs_f)[:6]
    assert(len(phrs) == len(ctxs) == len(tgts) == len(phr_tgt_sps) == len(cpts_tgt))
    return phrs, ctxs, tgts, cpts_tgt, phr_tgt_sps, stop_phrs


def main(args):
    chrf_t, n_sp_t = fmatch(args)
    print(f'Avg. chrF++:\t{chrf_t[0]:.3f} ± {chrf_t[1]:.3f}')
    print(f'Avg. spans:\t{n_sp_t[0]:.3f} ± {n_sp_t[1]:.3f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train', choices=['train', 'dev', 'test'])
    ap.add_argument('--data_dir', default='canard')
    ap.add_argument('--mmode', default='difflib', choices=['regex', 'difflib'])
    ap.add_argument('--tmode', default='bup', choices=['tdown', 'bup'])
    ap.add_argument('--stop_phrs_f', default='canard/stop_phrs.txt')
    ap.add_argument('--n_proc', type=int, default=2 * cpu_count() // 3)
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--print_found', action='store_true')
    ap.add_argument('--out_f', default='sps_{}_{}.json')
    args = ap.parse_args()
    main(args)
