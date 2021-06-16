#!/usr/bin/env python3
import argparse
import difflib
import json
import numpy as np
import regex

from collections import deque
from functools import partial
from nltk import Tree
from sacrebleu import sentence_chrf
from tqdm import tqdm

from proc_unmatch import _read_leaf


def rm_stop_phr(phr, stop_phrs):
    for prep in stop_phrs:
        if phr.startswith(prep) or phr.endswith(prep):
            phr = phr.replace(prep if len(phr) == len(prep) else prep + ' ', '')
    return phr


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


def get_offset(phr_tgt_sps, tgt, phr):
    """Find first token position of target matching phrase. Return split phrase
    and number of phrase tokens too."""
    tgt, phr = tgt.split(), phr.split()
    offset, pl
    return offset, pl, phr, tgt


def fill_word_bnd(ctx):
    ctx_spl = ctx.split()
    cind = -1
    word_bnd = {cind: 0}
    for i, w in enumerate(ctx_spl):  # map char to token indices of spaces
        cind += len(w) + 1
        word_bnd[cind] = len(word_bnd)
    word_bnd[len(ctx)] = len(word_bnd)
    return word_bnd


def fmatch_single(phr, ctx, tgt, cpt_tgt, phr_tgt_sp, match_fn, tmode):
    ctx = ' '.join(ctx.split(' [SEP] '))
    word_bnd = fill_word_bnd(ctx)
    #phr = rm_stop_phr(phr, stop_phrs)
    offset, pl = phr_tgt_sp
    phr_spl, tgt_spl = phr.split(), tgt.split()
    import pdb; pdb.set_trace()
    bspans = {}

    def tightest_span(t, i, k):
        if i > 0 or k < pl:
            return None
        sp_b = None
        i2 = i
        for st in t:
            if type(st) is Tree:
                k2 = len(st.leaves())
                o = tightest_span(st, i, k2)
                if o is not None:
                    sp_b = o
                    break
                i2 += k2
            else:
                i2 += 1
        return sp_b if sp_b else (t, i, k)

    def check_span(a2, n2):
        if a2 > -1 and a2 < len(ctx):
            sphr = ctx[a2:a2+n2]
            ek = a2 + n2 - 1
            while a2 > -1 and not ctx[a2].isspace():  # closest space index <= a2
                a2 -= 1
            while ek < len(ctx) and not ctx[ek].isspace():  # same >= a2 + n2
                ek += 1
            sp = (word_bnd[a2], word_bnd[ek])
            if ek - a2 < 2 * len(sphr):  # ignore bad matches
                return n2, sp
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
                    if n2 > 1:
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
        tup = tightest_span(cpt_tgt, i, k)
        bsp = None
        if tup is not None:
            t, i, k = tup
        else:
            t = cpt_tgt
        _ = trav(t, i)
        bsp = (i, k)
        return bsp

    def top_down():
        """Stop exploring child spans if their summed match count is less than
        parent's. This helps minimize number of tree splits.
        """
        def iter_children(t, i, k):
            spans, sts = [], []
            i2 = i
            cn = 0
            for st in t:
                if type(st) is Tree:
                    k2 = len(st.leaves())
                    if -k2 < i2 and i2 < pl:
                        a2, n2, sp = search_span(i2, k2)
                        itm = (i2, k2, a2, n2, sp)
                        bspans[(i2, k2)] = [itm]
                        spans.append(itm)
                        sts.append((st, i2, k2))
                        cn += n2
                    i2 += k2
                else:
                    i2 += 1
                if i2 >= pl:
                    break
            return spans, sts, cn

        i, k = -offset, len(cpt_tgt.leaves())
        bsp = None
        if offset > -1 and len(tgt_spl) == k:
            t, i, k = tightest_span(cpt_tgt, -offset, len(cpt_tgt.leaves()))
            bsp = (i, k)
            bspans[bsp] = [(i, k, *search_span(i, k))]
            dq = deque([(t, i, k)])
            while dq:
                t, i, k = dq.pop()
                a, n = bspans[(i, k)][0][2:-1]
                spans, sts, cn = iter_children(t, i, k)
                if cn > n:
                    bspans[(i, k)][:] = spans[:]
                    dq.extendleft(sts)
        return bsp

    bsp = bottom_up() if tmode == 'bup' else top_down()
    m, sps, n_sp = '', None, 0
    if bsp:
        m = ' '.join([ctx[a2:a2+n2] for _, _, a2, n2, _ in bspans[bsp]])
        m = regex.sub('^\s+|\s+$|\s+(?=\s)', '', m)
        sps = [t[-1] for t in bspans[bsp]]
        n_sp = len(bspans[bsp])
    return m, phr, sps, n_sp


def fmatch(args):
    phrs, ctxs, tgts, cids, cpts, cpts_tgt, phr_tgt_sps = read_fs(args)

    match_fn = regex_match if args.mmode == 'regex' else difflib_match
    fms = partial(fmatch_single, match_fn=match_fn, tmode=args.tmode)
    #res = [fms(*t) for t in tqdm(zip(phrs, ctxs, tgts, cpts_tgt, phr_tgt_sps))]
    res = [fmt(phrs[i][j], ctxs[i], tgts[i], cpts_tgt[i], phr_tgt_sps[i][j])\
            for i in range(len(phrs)) for j in range(len(phrs[i]))]

    with open(args.out_f.format(args.mmode, args.tmode), 'w', encoding='utf8') as f:
        json.dump([t[2] for t in res], f)
    chrfs = np.array([sentence_chrf(m, [p]).score if m else 0. for m, p, _, _ in res])
    n_sps = np.array([t[3] for t in res])
    return (chrfs.mean(), chrfs.std()), (n_sps.mean(), n_sps.std())


def read_fs(args):
    """Returns (phrs, ctxs, tgts, phr tree IDs, unique phr trees, tgt trees)."""
    with open(args.lems_f, encoding='utf8') as f:
        lems = json.load(f)
        ctxs, tgts = lems[::2], lems[1::2]
    with open(args.phr_tgt_sps_f, encoding='utf8') as f:
        phr_tgt_sps = json.load(f)
        phrs = [[ctxs[i].split()[t[0]:t[0]+t[1]] for t in l2] for i, l2 in enumerate(phr_tgt_sps)]
    with open(args.cids_f, encoding='utf8') as f:
        cids = {i: l.rstrip() for i, l in enumerate(f)}
    with open(args.cpts_f, encoding='utf8') as f:
        cpts = [Tree.fromstring(l.rstrip(), read_leaf=_read_leaf) for l in f]
    with open(args.cpts_tgt_f, encoding='utf8') as f:
        cpts_tgt = [Tree.fromstring(l.rstrip(), read_leaf=_read_leaf) for l in f]
    with open(args.stop_phrs_f) as f:
        stop_phrs = [phr.strip() for phr in f.readlines()[:7]]
    assert(len(phrs) == len(ctxs) == len(tgts) == len(cids) == len(cpts_tgt))
    return phrs, ctxs, tgts, cids, cpts, cpts_tgt, phr_tgt_sps


def main(args):
    chrf_t, n_sp_t = fmatch(args)
    print(f'Avg. chrF++:\t{chrf_t[0]:.4f} ± {chrf_t[1]:.4f}')
    print(f'Avg. spans:\t{n_sp_t[0]:.4f} ± {n_sp_t[1]:.4f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mmode', default='difflib', choices=['regex', 'difflib'])
    ap.add_argument('--tmode', default='bup', choices=['tdown', 'bup'])
    ap.add_argument('--lems_f', default='canard/unmatch_lems.json')
    ap.add_argument('--cids_f', default='canard/cpt_ids.txt')
    ap.add_argument('--cpts_f', default='canard/cpts_uniq.txt')
    ap.add_argument('--cpts_tgt_f', default='canard/cpts_tgt.txt')
    ap.add_argument('--stop_phrs_f', default='canard/stop_phrs.txt')
    ap.add_argument('--phr_tgt_sps_f', default='canard/phr_tgt_sps.json')
    ap.add_argument('--out_f', default='canard/sps_{}_{}.json')
    args = ap.parse_args()
    main(args)