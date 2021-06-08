#!/usr/bin/env python3
import argparse
import difflib
import json
import regex

from nltk import Tree
from sacrebleu import sentence_chrf

PREP = set(['besides', 'other than', 'aside from', 'in addition to'])


def rm_prep(phr):
    for prep in PREP:
        if phr.find(prep) > -1:
            phr = phr.replace(prep if len(phr) == len(prep) else prep + ' ', '')
            break
    return phr


def trav(t, s, ctx, bspans, offset, phr_spl, pl):
    """Iterate through tree spans. Return (i, j - i)."""
    a = -1  # start character index in context match
    n = 0  # number of matched characters in context
    i = s - offset  # span token index in phrase
    k = len(t.leaves())  # span token length in phrase
    if s < offset + pl:
        cur_phr = ' '.join(phr_spl[i:i+k])
        ms = difflib.SequenceMatcher(None, ctx, cur_phr).get_matching_blocks()
        a, _, n = max(ms, key=lambda m: m.size)
    if i < pl:
        cn = 0
        spans = []
        for st in t:
            if type(st) is Tree:
                s, a2, n2, i2, k2 = trav(st, s, ctx, bspans, offset, phr_spl, pl)
                if i > -1:
                    spans.append((i2, k2, a2, n2))
                    cn += n2
            else:
                s += 1
        bspans[(i, k)] = spans if cn > n else [(i, k, a, n)]
    return s, a, n, i, k


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


def difflib_match(phr, ctx, tgt, cpt_tgt):
    phr = rm_prep(phr)
    bspans = {}
    offset, phr_spl, pl = get_offset(tgt, phr)
    _, _, _, i, k = trav(cpt_tgt, offset, ctx, bspans, offset, phr_spl, pl)
    m = ''
    if (i, k) in bspans:
        m = ' '.join([ctx[a2:a2+n2] for (i2, k2, a2, n2) in bspans[(i, k)] if n2 > 1])
    return m, phr


def regex_match(phr, ctx, tgt, cpt_tgt):
    ret = None
    try:
        phr = rm_prep(phr)
        # (?b) is BESTMATCH, (?e) is ENHANCEMATCH flag
        # NOTE: n.fuzzy_counts is (sub, ins, del) count
        m = regex.match(f'(?b)({phr})' + '{e}', ctx)
        ret = m[0]
    except regex._regex_core.error:  # no match
        pass
    return (ret, phr)


def fmatch(args):
    """Fuzzy matching using https://pypi.org/project/regex/"""
    phrs, ctxs, tgts, cids, cpts, cpts_tgt = read_fs(args)

    vld_i = (i for i, phr in enumerate(phrs) if len(phr) > 1)
    match_fn = regex_match if args.mode == 'regex' else difflib_match
    res = [match_fn(phrs[i], ctxs[i], tgts[i], cpts_tgt[i]) for i in vld_i]
    avg_chrf = sum([sentence_chrf(m, [p]).score if m else 0. for m, p in res])\
            / len(res)
    return avg_chrf


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
    avg_chrf = fmatch(args)
    print(f'Avg. chrF++: {avg_chrf:.4f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='regex', choices=['regex', 'difflib'])
    ap.add_argument('--lems_f', default='unmatch_lems2.json')
    ap.add_argument('--cids_f', default='pt_ids.txt')
    ap.add_argument('--cpts_f', default='pts_uniq.txt')
    ap.add_argument('--cpts_tgt_f', default='pts_tgt.txt')
    args = ap.parse_args()
    main(args)
