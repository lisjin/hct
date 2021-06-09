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

PREP = set(['besides', 'other than', 'aside from', 'in addition to'])


def rm_prep(phr):
    for prep in PREP:
        if phr.find(prep) > -1:
            phr = phr.replace(prep if len(phr) == len(prep) else prep + ' ', '')
            break
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


def get_offset(tgt, phr):
    """Find first token position of target matching phrase. Return split phrase
    and number of phrase tokens too."""
    offset = -1
    tgt, phr = tgt.split(), phr.split()
    if phr:
        pl = len(phr)
        for i in (j for j, x in enumerate(tgt) if x == phr[0]):
            if tgt[i:i+pl] == phr:
                offset = i
    return offset, phr, pl if offset > -1 else 0


def fmatch_single(phr, ctx, tgt, cpt_tgt, match_fn, tmode):
    phr = rm_prep(phr)
    offset, phr_spl, pl = get_offset(tgt, phr)
    bspans = {}

    def search_span(i, k, phr_spl, ctx):
        kp = k if i > -1 else k + i
        a, n = -1, 0
        if kp > 0:
            ip = max(0, i)
            cur_phr = ' '.join(phr_spl[ip:ip+kp])
            a, n = match_fn(ctx, cur_phr)
        return a, n

    def trav(t, s):
        """Compare tree span matches bottom-up, replacing parent score by
        children's as necessary."""
        a = -1  # start character index in context match
        n = 0  # number of matched characters in context
        i = s - offset  # span token index in phrase
        k = len(t.leaves())  # span token length in phrase
        if (i, k) not in bspans:
            a, n = search_span(i, k, phr_spl, ctx)
            cn = 0
            spans = []
            for st in t:
                if type(st) is Tree:
                    s, a2, n2, i2, k2 = trav(st, s)
                    if i > -1 and n2 > 1:
                        spans.append((i2, k2, a2, n2))
                        cn += n2
                else:
                    s += 1
            bspans[(i, k)] = [(i, k, a, n)]
            if cn > n:
                bspans[(i, k)][:] = spans[:]
                n = cn
        return s, a, n, i, k

    def bottom_up():
        _, _, _, i, k = trav(cpt_tgt, offset)
        m = ' '.join([ctx[a2:a2+n2] for (_, _, a2, n2) in bspans[(i, k)]])
        return m, (i, k)

    def top_down():
        """Stop exploring child spans if their summed match count is less than
        parent's. This helps minimize number of tree splits.
        """
        m = ''
        bsp = (-offset, len(cpt_tgt.leaves()))
        if offset > -1:
            def t_score(t, i):
                k = len(t.leaves())
                a, n = search_span(i, k, phr_spl, ctx)
                return a, n, k

            dq = deque([(cpt_tgt, -offset)])
            while dq:
                t, i = dq.pop()
                if type(t) is Tree:
                    a, n, k = t_score(t, i)
                    spans, sts = [], []
                    find_bsp = i <= 0 and k >= pl
                    i2 = i
                    cn = 0
                    for st in t:
                        if type(st) is Tree:
                            a2, n2, k2 = t_score(st, i2)
                            if n2 > 1:
                                spans.append((i2, k2, a2, n2))
                                sts.append((st, i2))
                                cn += n2
                                if find_bsp and i2 <= 0 and i2 + k2 >= pl:
                                    bsp = (i2, k2)
                                    cn = n2
                                    spans = spans[-1:]
                                    sts = sts[-1:]
                                    break
                            i2 += k2
                            if i2 >= pl:
                                break
                    bspans[(i, k)] = [(i, k, a, n)]
                    if cn > n or find_bsp and cn == n:
                        bspans[(i, k)][:] = spans[:]
                        dq.extendleft(sts)

            m = ' '.join([ctx[a2:a2+n2] for (_, _, a2, n2) in bspans[bsp]])
        return m, bsp

    m, bsp = bottom_up() if tmode == 'bup' else top_down()
    m = regex.sub('^\s+|\s+$|\s+(?=\s)', '', m)  # remove lead/trail/multi space
    return m, phr, len(bspans[bsp]) if bsp in bspans else 0


def fmatch(args):
    phrs, ctxs, tgts, cids, cpts, cpts_tgt = read_fs(args)

    match_fn = regex_match if args.mmode == 'regex' else difflib_match
    fms = partial(fmatch_single, match_fn=match_fn, tmode=args.tmode)
    vld_set = (t for t in zip(phrs, ctxs, tgts, cpts_tgt) if len(t[0]) > 1)
    res = [fms(*t) for t in tqdm(vld_set)]

    chrfs = np.array([sentence_chrf(m, [p]).score if m else 0. for m, p, _ in res])
    n_sps = np.array([n_sp for _, _, n_sp in res])
    return (chrfs.mean(), chrfs.std()), (n_sps.mean(), n_sps.std())


def read_fs(args):
    """Returns (phrs, ctxs, tgts, phr tree IDs, unique phr trees, tgt trees)."""
    with open(args.lems_f, 'r', encoding='utf8') as f:
        lems = json.load(f)
        phrs, ctxs, tgts = lems[::3], lems[1::3], lems[2::3]
    with open(args.cids_f, encoding='utf8') as f:
        cids = {i: l.rstrip() for i, l in enumerate(f)}
    with open(args.cpts_f, encoding='utf8') as f:
        cpts = [Tree.fromstring(l.rstrip()) for l in f]
    with open(args.cpts_tgt_f, encoding='utf8') as f:
        cpts_tgt = [Tree.fromstring(l.rstrip()) for l in f]
    assert(len(phrs) == len(ctxs) == len(tgts) == len(cids) == len(cpts_tgt))
    return phrs, ctxs, tgts, cids, cpts, cpts_tgt


def main(args):
    chrf_t, n_sp_t = fmatch(args)
    print(f'Avg. chrF++:\t{chrf_t[0]:.4f} ± {chrf_t[1]:.4f}')
    print(f'Avg. spans:\t{n_sp_t[0]:.4f} ± {n_sp_t[1]:.4f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mmode', default='difflib', choices=['regex', 'difflib'])
    ap.add_argument('--tmode', default='bup', choices=['tdown', 'bup'])
    ap.add_argument('--lems_f', default='unmatch_lems2.json')
    ap.add_argument('--cids_f', default='pt_ids.txt')
    ap.add_argument('--cpts_f', default='pts_uniq.txt')
    ap.add_argument('--cpts_tgt_f', default='pts_tgt.txt')
    args = ap.parse_args()
    main(args)
