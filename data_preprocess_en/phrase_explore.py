#!/usr/bin/env python3
import argparse
import json

from collections import Counter
from itertools import chain
from nltk import Tree
from os import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool

from utils import read_expand, fromstring


def read_fs(args):
    with Pool(args.n_proc) as p:
        with open(args.cpts_uniq_f, encoding='utf8') as f:
            cpts_uniq = p.map(fromstring, [l.rstrip() for l in f])
    with open(args.cids_f, encoding='utf8') as f:
        cids = [list(map(int, l.rstrip().split(','))) for l in f]
    cids, _, _, _, _ = read_expand(args.phr_tgt_sps_f, cpts_tgt=cids)
    with open(args.ctx_sps_f) as f:
        ctx_sps = json.load(f)
    assert(len(cids) == len(ctx_sps))
    return cpts_uniq, cids, ctx_sps


def find_subspans(t, i, k, pl, li, sps):
    i2 = i
    for st in t:
        k2 = len(st.leaves()) if type(st) is Tree else 1
        if i2 == li and k2 <= pl:
            sps.append((i2, k2))
            li += k2
        elif i2 + k2 > li and type(st) is Tree:
            li = find_subspans(st, i2, k2, pl, li, sps)
        i2 += k2
        if i2 >= pl:
            break
    return li


def get_sps(pts_ctx, sp):
    offset, pl = sp[0], sp[1] - sp[0]
    i = -offset
    sps = []
    for pt in pts_ctx:
        k = len(pt.leaves())
        if i + k > 0:
            sps2 = []
            pl -= find_subspans(pt, i, k, pl, 0, sps2)
            sps.extend(sps2[:])
            if pl > 0:
                i = 0
            else:
                break
        else:
            i += k
    return sps


def main(args):
    cpts_uniq, cids, ctx_sps = read_fs(args)
    hist = Counter()
    for i, cid in enumerate(cids):
        phr_sps = []
        pts_ctx = [cpts_uniq[c] for c in cid]
        for sp in ctx_sps[i]:  # context subspans for current phrase
            phr_sps.extend(get_sps(pts_ctx, sp))
        hist[len(phr_sps)] += 1
    print('# sps\tfreq')
    for k in sorted(hist.keys()):
        print(k, hist[k], sep='\t')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phr_tgt_sps_f', default='canard/phr_tgt_sps.json')
    ap.add_argument('--unmatch_path', default='canard/unfound_phrs.json')
    ap.add_argument('--cids_f', default='canard/cpt_ids.txt')
    ap.add_argument('--cpts_uniq_f', default='canard/cpts_uniq.txt')
    ap.add_argument('--ctx_sps_f', default='canard/sps_difflib_tdown.json')
    ap.add_argument('--n_proc', type=int, default=2 * cpu_count() // 3)
    args = ap.parse_args()
    main(args)
