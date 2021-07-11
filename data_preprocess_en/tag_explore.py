#!/usr/bin/env python3
import argparse
import json
import os

from collections import Counter
from itertools import chain
from nltk import Tree
from os import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool

from utils import fromstring, concat_path, find_subspans


def maybe_int(x):
    if ',' in x:
        return tuple(map(int, x.split(',')))
    else:
        return int(x)


def load_tags(args):
    tag2idx = {x: i for i, x in enumerate(['KEEP', 'DELETE'])}
    tags, st_ens = [], []
    with open(concat_path(args, 'tags.txt', data_out=True), encoding='utf8') as f:
        for l in f:
            tags.append([])
            st_ens.append([])
            for x in l.rstrip().split():
                t, se = x.split('|')
                tags[-1].append(tag2idx[t])
                st_ens[-1].append(tuple(map(maybe_int, se.split('#'))))
    return tags, st_ens


def read_fs(args):
    if args.split == 'train':
        with open(concat_path(args, 'cnv_ids.json')) as f:
            cnv_ids = set(json.load(f))
    with Pool(args.n_proc) as p:
        with open(concat_path(args, 'cpts_uniq1.txt'), encoding='utf8') as f:
            cpts_uniq = p.map(fromstring, [l.rstrip() for l in f])
    with open(concat_path(args, 'cpt_ids1.txt'), encoding='utf8') as f:
        cids = [list(map(int, l.rstrip().split(','))) for l in f]
        with open(concat_path(args, 'cpts_src.txt'), encoding='utf8') as f:
            cpts_src = p.map(fromstring, [l.rstrip() for l in f])
        if args.split == 'train':
            cids[:] = [cid for i, cid in enumerate(cids) if i in cnv_ids]
            cpts_src[:] = [pt for i, pt in enumerate(cpts_src) if i in cnv_ids]
    tags, st_ens = load_tags(args)
    assert(len(cpts_src) == len(tags))
    return cpts_uniq, cids, cpts_src, tags, st_ens


def iter_pt(pts, pi, pi_max, pl, li, sps):
    while pl > 0 and pi < pi_max:
        pt = pts[pi]
        k = len(pt.leaves())
        pl2 = min(k, pl)
        _ = find_subspans(pt, 0, k, pl2, li, sps)
        mv_pt = pl >= k
        pl -= (pl2 - li)
        if mv_pt:
            pi += 1
            li = 0
        else:
            li = pl
    return pi, li


def get_tag_sps(pts, tag, sps):
    offset, en, li, n = 0, 0, 0, len(tag)
    pi, pi_max = 0, len(pts)
    sps.clear()
    while offset < n:
        while en < n and tag[en] == tag[offset]:
            en += 1
        pl = en - offset
        pi, li = iter_pt(pts, pi, pi_max, pl, li, sps)
        offset = en


def get_se_sps(pts, st_en, sps):
    lptr, li, pl_tot = 0, 0, 0
    pi, pi_max = 0, len(pts)
    sps.clear()
    for s, e in st_en:
        if type(s) is tuple or s > -1:
            if type(s) is int:
                s, e = (s,), (e,)
            for s2, e2 in zip(s, e):
                pi, lptr = 0, 0
                while pi < pi_max:
                    k = len(pts[pi].leaves())
                    if lptr + k > s2:
                        break
                    lptr += k + 1  # account for delimiter token
                    pi += 1
                pl = e2 - s2
                pi2, _ = iter_pt(pts, pi, pi_max, pl, li, sps)
                pl_tot += pl
    if sps:
        spl_tot = sum(t[1] for t in sps)
        assert(spl_tot == pl_tot or pl_tot > spl_tot)


def print_hist(hist):
    print('# sps\tfreq')
    for k in sorted(hist.keys()):
        print(k, hist[k], sep='\t')


def main(args):
    cpts_uniq, cids, cpts_src, tags, st_ens = read_fs(args)
    hist_tag, hist_se = Counter(), Counter()
    sps_tag, sps_se = [], []
    for i2, (cid, pt_src, tag, st_en) in enumerate(zip(cids, cpts_src, tags, st_ens)):
        pts_ctx = [cpts_uniq[c] for c in cid]
        pts_inp = pts_ctx[:] + [pt_src]
        dlm_is = set()  # delimiter ([SEP], [CI]) indices to ignore
        if len(pts_inp) > 1:
            j = 0
            for pt in pts_inp[:-1]:
                j += len(pt.leaves()) + 1
                dlm_is.add(j)

        # Ignore last '*' character in tag
        tag[:] = [t for j, t in enumerate(tag[:-1]) if j not in dlm_is]
        tot_l = sum(len(pt.leaves()) for pt in pts_inp)
        assert(tot_l == len(tag))

        get_tag_sps(pts_inp, tag, sps_tag)
        assert(sum(t[1] for t in sps_tag) == tot_l)
        hist_tag[len(sps_tag)] += 1

        get_se_sps(pts_ctx, st_en, sps_se)
        hist_se[len(sps_se)] += 1
    print_hist(hist_tag)
    print_hist(hist_se)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train', choices=['train', 'dev', 'test'])
    ap.add_argument('--data_dir', default='canard')
    ap.add_argument('--data_out_dir', default='canard_out')
    ap.add_argument('--n_proc', type=int, default=min(4, cpu_count()))
    args = ap.parse_args()
    main(args)
