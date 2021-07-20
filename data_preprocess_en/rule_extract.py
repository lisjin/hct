#!/usr/bin/env python3
import argparse
import json
import numpy as np
import os

from collections import Counter
from functools import partial
from itertools import chain
from nltk import Tree
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import AffinityPropagation

from fmatch_lem import fmatch_single
from utils import fromstring, read_lem, read_expand, concat_path
from utils_data import yield_sources_and_targets


punct_set=set(['!', ',', '?'])


class SlottedRule:
    def __init__(self, tokens, slot_spans, validate=False, mask='_', ignore_trail_punct=True):
        self.tokens = tokens
        self.slot_spans = [(max(0, i), min(i + k, len(self))) for i, k in slot_spans]
        self.n_slots = len(self.slot_spans)
        self.term_spans, self.n_term_spans, self.n_terms = self.get_term(slot_spans)

        self.ignore_trail_punct = ignore_trail_punct
        self.mask = mask
        if validate:
            self.validate()

    def __len__(self):
        return len(self.tokens)

    def __str__(self):
        rule_str = f' {self.mask} '.join([' '.join(l) for l in self.term_tokens])
        if len(self.term_spans):
            pref = f'{self.mask} ' if self.term_spans[0][0] > 0 else ''
            suf = f' {self.mask}' if self.term_spans[-1][1] < len(self) else ''
            return f'{pref}{rule_str}{suf}'
        else:
            return ' '.join([self.mask] * len(self.slot_spans))

    @property
    def slot_tokens(self):
        return self.span_tokens(self.slot_spans)

    @property
    def term_tokens(self):
        ret = self.span_tokens(self.term_spans)
        if self.ignore_trail_punct and ret and ret[-1][-1] in punct_set:
            ret[-1][:] = ret[-1][:-1]
        return ret

    def span_tokens(self, spans):
        return [self.tokens[s:e] for s, e in spans]

    def adv_span(self, spans, i):
        while i < len(spans) - 1 and spans[i][1] == spans[i + 1][0]:
            i += 1
        if i < len(spans):
            lptr = spans[i][1]
            rptr = lptr + 1
        else:
            lptr, rptr = spans[i - 1][1], len(self)
        return i + 1, lptr, rptr

    def get_term(self, slot_spans):
        sps = []
        i, lptr, rptr, n_terms = 0, 0, 0, 0
        if self.slot_spans and self.slot_spans[i][0] == 0:
            i, lptr, rptr = self.adv_span(self.slot_spans, i)
        while i < self.n_slots:
            while rptr < self.slot_spans[i][0] and rptr < len(self):
                rptr += 1
                n_terms += 1
            sps.append((lptr, rptr))
            i, lptr, rptr = self.adv_span(self.slot_spans, i)

        if lptr < len(self):
            sps.append((lptr, len(self)))
        return sps, len(sps), n_terms

    def validate(self):
        s1 = set(chain.from_iterable(range(*sp) for sp in self.slot_spans +\
                self.term_spans))
        assert(s1 == set(range(len(self))))


def lcs_dist(r1, r2):
    """Longest common subsequence distance over tokens in strings r1, r2."""
    t1, t2 = str(r1).split(), str(r2).split()
    if len(t1) > len(t2):
        t1, t2 = t2, t1
    dists = range(len(t1) + 1)
    for i2, c2 in enumerate(t2):
        dists_ = [i2 + 1]
        for i1, c1 in enumerate(t1):
            if c1 == c2:
                dists_.append(dists[i1])
            else:
                dists_.append(min(dists_[-1], dists[i1 + 1]) + 1)
        dists = dists_
    return dists[-1] / float(len(t1) + len(t2))


def exact_match(ctx, phr):
    a = ctx.find(phr)
    if a > -1:
        return a, len(phr)
    return -1, 0


def read_fs(args):
    with Pool(args.n_proc) as p:
        with open(concat_path(args, 'cpts_tgt.txt'), encoding='utf8') as f:
            cpts_tgt = p.map(fromstring, [l.rstrip() for l in f])
    ctxs, tgts = [], []
    for source, target in yield_sources_and_targets(os.path.join(args.data_dir, f'{args.split}.tsv'), args.tsv_fmt):
        ctxs.append(source[0].split(' [CI] ')[0])
        tgts.append(target)
    cpts_tgt, ctxs, tgts, phrs, phr_tgt_sps = read_expand(concat_path(args,
        'phr_tgt_sps.json'), ctxs=ctxs, tgts=tgts, cpts_tgt=cpts_tgt)
    phr_tgt_sps = [sp for pts in phr_tgt_sps for sp in pts]
    assert(len(phrs) == len(ctxs) == len(tgts) == len(phr_tgt_sps) == len(cpts_tgt))
    return phrs, ctxs, tgts, cpts_tgt, phr_tgt_sps


def get_slot_spans(r):
    return [r2[:2] for r2 in r[1]]


def count_rules(srs):
    rules = Counter()
    for sr in srs:
        rules[str(sr)] += 1
    nr = len(rules)
    print(f'Found {nr} rules')
    return rules, nr


def get_triu_dist(rstrs, nr):
    """Return flat upper-triangular distance between strings.
    Args:
        rstrs: List of rule strings of length nr
        nr: Number of rules
    """
    tups = ((rstrs[i], rstrs[j]) for i in range(nr) for j in range(i + 1, nr))
    with Pool(args.n_proc) as p:
        triu_dist = p.map(lcs_dist, *zip(*tups))
    return triu_dist


def triu_to_full(triu_dist, nr):
    """Convert flat to full distance matrix.
    Args:
        triu_dist: Flat upper-triangular distance matrix
        nr: Number of rules
    """
    dist = np.empty((nr, nr))
    triu_mask = np.triu(np.ones((nr, nr), dtype=bool))
    np.fill_diagonal(triu_mask, 0)
    dist[triu_mask] = triu_dist
    dist.T[triu_mask] = triu_dist
    return dist


def main(args):
    phrs, ctxs, tgts, cpts_tgt, phr_tgt_sps = read_fs(args)
    ems = partial(fmatch_single, match_fn=exact_match, tmode='bup')
    if not args.debug:
        with Pool(args.n_proc) as p:
            res = p.map(ems, phrs, ctxs, cpts_tgt, phr_tgt_sps)
            srs = p.map(SlottedRule, phrs, [get_slot_spans(r) for r in res])
    else:
        res = [ems(*t) for t in list(zip(phrs, ctxs, cpts_tgt, phr_tgt_sps))]
        srs = [SlottedRule(phr, get_slot_spans(r)) for phr, r in zip(phrs, res)]

    rules, nr = count_rules(srs)
    rstrs = list(rules.keys())
    triu_dist = get_triu_dist(rstrs, nr)
    if args.cluster_method == 'affinity':
        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html
        dist = triu_to_full(triu_dist, nr)
        af = AffinityPropagation(affinity='precomputed', random_state=None).fit(-dist)
        labels = af.labels_
    elif args.cluster_method == 'hierarch':
        # https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
        Z = linkage(triu_dist, method='ward')
        dist_thresh = np.percentile(triu_dist, args.perc_q)
        labels = fcluster(Z, t=dist_thresh, criterion='distance')

    clusters = {}
    for k, v in enumerate(labels):
        clusters.setdefault(v, []).append(k)
    print(f'Reduced to {len(clusters)} rules')
    with open(concat_path(args, args.out_f.format(args.cluster_method)), 'w', encoding='utf8') as f:
        json.dump([[rstrs[j] for j in v] for v in clusters.values()], f, indent=4)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train', choices=['train', 'dev', 'test'])
    ap.add_argument('--data_dir', default='canard')
    ap.add_argument('--n_proc', type=int, default=min(4, os.cpu_count()))
    ap.add_argument('--tsv_fmt', default='wikisplit')
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--out_f', default='rule_{}.json')
    ap.add_argument('--perc_q', type=int, default=10, help='For --cluster_method=hierarch: percentile of pairwise distance to bound inter-cluster distance')
    ap.add_argument('--cluster_method', default='affinity', choices=['affinity', 'hierarch'])
    args = ap.parse_args()
    main(args)
