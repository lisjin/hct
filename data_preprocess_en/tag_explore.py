#!/usr/bin/env python3
import argparse
import json
import numpy as np
import os

from collections import Counter
from itertools import chain
from nltk import Tree
from os import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool

from utils import fromstring, concat_path, find_subspans, write_lst

tag_lst = ['KEEP', 'DELETE']


def maybe_int(x):
    if ',' in x:
        return tuple(map(int, x.split(',')))
    else:
        return int(x)


def load_tags(args):
    global tag_lst
    tag2idx = {x: i for i, x in enumerate(tag_lst)}
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


def fpost2str(fpost):
    if type(fpost) is tuple:
        return ','.join(str(x) for x in fpost)
    return str(fpost)


def write_tags(args, st_ens, sps_tag_all, tag_s_all):
    global tag_lst
    line, lines = [], []
    for st_en, sp_tag, tag_s in zip(st_ens, sps_tag_all, tag_s_all):
        line.clear()
        src_i = tag_s
        for offset, en, action in sp_tag:
            s, e = st_en[src_i]
            line.append(f'{tag_lst[action]}|{fpost2str(s)}#{fpost2str(e)}|{offset}#{en}')
        lines.append(' '.join(line))
    write_lst(concat_path(args, 'tags1.txt', data_out=True), lines)


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


def iter_pts(pts, pi, pi_max, pl, li, sps, pl_tot):
    while pl > 0 and pi < pi_max:
        pt = pts[pi]
        k = len(pt.leaves())
        pl2 = min(k, pl)
        _ = find_subspans(pt, 0, k, li + pl2, li, sps)
        mv_pt = pl > k
        pl -= pl2
        if mv_pt:
            pi += 1
            li = 0
        else:
            li += pl2
        pl_tot += pl2
        pl -= pl2 + 1
        pi += 1
    return pl_tot, li


def check_sps(sps, pl_tot):
    if sps:
        spl_tot = sum(t[1] for t in sps)
        assert(spl_tot == pl_tot)


def get_tag_sps(pts, tag, insert_ind, sps, hist):
    offset, en, li, pl_tot, n = 0, 0, 0, 0, len(tag)
    pi, pi_max = 0, len(pts)
    sps2 = []
    sps.clear()
    while offset < n:
        while en < n and tag[en] == tag[offset]:
            if en != offset and en in insert_ind:
                break
            en += 1
        pl = en - offset
        sps2.clear()
        pl_tot, li = iter_pts(pts, pi, pi_max, pl, li, sps2, pl_tot)
        hist[len(sps2)] += 1
        sps.extend((*sp, tag[offset]) for sp in sps2)
        offset = en
    check_sps(sps, pl_tot)


def get_se_sps(pts, st_en, sps, hist):
    lptr, li, pl_tot = 0, 0, 0
    pi, pi_max = 0, len(pts)
    sps2 = []
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
                sps2.clear()
                pl = e2 - s2 + 1
                pl_tot, _ = iter_pts(pts, pi, pi_max, pl, li, sps2, pl_tot)
                hist[len(sps2)] += 1
                sps.extend(sps2)
    check_sps(sps, pl_tot)


def print_hist(hist, fmt_str='{:d}'):
    print('# sps\tfreq')
    for k in sorted(hist.keys()):
        print(fmt_str.format(k), hist[k], sep='\t')


def main(args):
    cpts_uniq, cids, cpts_src, tags, st_ens = read_fs(args)
    hist_tag, hist_se = Counter(), Counter()
    sps_tag, sps_se = [], []
    if args.write:
        sps_tag_all, tag_s_all = [], []
    for cid, pt_src, tag, st_en in zip(cids, cpts_src, tags, st_ens):
        pts_src = [pt_src]
        pts_ctx = [cpts_uniq[c] for c in cid] + pts_src

        insert_ind = set(i for i, x in enumerate(st_en) if type(x[0]) is tuple\
                or x[0] > -1)
        tag_s = len(tag) - len(pt_src.leaves())
        get_tag_sps(pts_src, tag[tag_s:], insert_ind, sps_tag, hist_tag)
        if args.write:
            sps_tag_all.append(sps_tag[:])
            tag_s_all.append(tag_s)
        get_se_sps(pts_ctx, st_en, sps_se, hist_se)
    if args.print_stats:
        print_hist(hist_tag)
        print_hist(hist_se)
    if args.write:
        write_tags(args, st_ens, sps_tag_all, tag_s_all)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train', choices=['train', 'dev', 'test'])
    ap.add_argument('--data_dir', default='canard')
    ap.add_argument('--data_out_dir', default='canard_out')
    ap.add_argument('--n_proc', type=int, default=min(4, cpu_count()))
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--print_stats', action='store_true')
    args = ap.parse_args()
    main(args)
