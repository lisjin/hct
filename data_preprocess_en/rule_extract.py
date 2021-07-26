#!/usr/bin/env python3
import argparse
import json
import numpy as np
import os

from collections import Counter
from functools import partial
from itertools import chain
from math import floor
from nltk import Tree
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import AffinityPropagation

from fmatch_lem import fmatch_single
from utils import fromstring, read_lem, read_expand, concat_path, write_lst
from utils_data import filter_sources_and_targets


punct_set = set([',', '?', '!'])


class SlottedRule:
    def __init__(self, tokens, orig_tokens, slot_spans, max_sp_width, ignore_trail_punct, mask='_', validate=False):
        self.tokens = tokens
        self.orig_tokens = orig_tokens
        self.slot_spans = [(max(0, i), min(i + k, len(self))) for i, k in slot_spans]
        self.n_slots = len(self.slot_spans)
        self.term_spans, self.n_term_spans, self.n_terms = self.get_term(slot_spans)

        self.max_sp_width = max_sp_width
        self.ignore_trail_punct = ignore_trail_punct
        self.mask = mask
        if validate:
            self.validate()

    def __len__(self):
        return len(self.tokens)

    def __str__(self, use_lemma=True):
        if len(self.term_spans):
            pref = f'{self.mask} ' if self.term_spans[0][0] > 0 else ''
            suf = f' {self.mask}' if self.term_spans[-1][1] < len(self) else ''
            n_term_join = self.max_sp_width - int(len(pref) > 0) - int(len(suf) > 0) + 1
            term_phrs = [' '.join(l) for l in (self.term_tokens if use_lemma\
                    else self.term_orig_tokens)]
            if term_phrs:
                rule_str = f' {self.mask} '.join(term_phrs[:n_term_join])
                return f'{pref}{rule_str}{suf}'
        return ' '.join([self.mask] * min(len(self.slot_spans), self.max_sp_width))

    @property
    def slot_tokens(self):
        return self.span_tokens(self.slot_spans)

    @property
    def term_tokens(self):
        ret = self.span_tokens(self.term_spans)
        if self.ignore_trail_punct and ret and ret[-1][-1] in punct_set:
            ret[:] = ret[:-1]
        return ret

    @property
    def term_orig_tokens(self):
        return [self.orig_tokens[s:e] for s, e in self.term_spans]

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

    phr_tgt_sps_f = concat_path(args, 'phr_tgt_sps.json')
    ctxs, tgts = read_lem(concat_path(args, 'unmatch_lems.json'))
    cpts_tgt, ctxs, tgts, phrs, phr_tgt_sps = read_expand(phr_tgt_sps_f,
            ctxs=ctxs, tgts=tgts, cpts_tgt=cpts_tgt)
    phr_tgt_sps = [sp for pts in phr_tgt_sps for sp in pts]

    tgts_orig = []
    with open(concat_path(args, 'unfound_phrs.json'), encoding='utf8') as f:
        outs_k = set([int(k) for k in json.load(f).keys()])
    tgts_orig = [target for _, _, target in filter_sources_and_targets(
        os.path.join(args.data_dir, f'{args.split}.tsv'), outs_k)]
    _, _, _, phrs_orig, _ = read_expand(phr_tgt_sps_f, tgts=tgts_orig)

    assert(len(phrs) == len(ctxs) == len(tgts) == len(phrs_orig) == len(phr_tgt_sps) == len(cpts_tgt))
    return phrs, ctxs, tgts, phrs_orig, cpts_tgt, phr_tgt_sps


def get_slot_spans(r):
    return [r2[:2] for r2 in r[1]]


def count_rules(srs):
    """Rule frequency across data instances.
    Returns:
        rules: Counter object keyed by rule strings
        rstr_i: NumPy array of size len(srs) with rule strings per data instance
        rstr2orig: dictionary mapping between lemmatized and original rule
        nr: Number of unique rule strings
    """
    rules = Counter()
    rstr2orig = {}
    rstr_i = []
    for i, sr in enumerate(srs):
        rstr = str(sr)
        if rstr:
            if rstr not in rules:
                rstr2orig[rstr] = (len(rstr2orig), sr.__str__(use_lemma=False))
            rstr_i.append(rstr2orig[rstr][0])
            rules[rstr] += 1
        else:
            import pdb; pdb.set_trace()
            rstr_i.append(-1)
    nr = len(rules)
    rstr2orig = {k: v[1] for k, v in rstr2orig.items()}
    print(f'Found {nr} rules')
    return rules, np.array(rstr_i), rstr2orig, nr


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


def group_triu_dist(triu_dist, nr):
    ret = []
    lptr = 0
    for rlen in range(nr - 1, -1, -1):
        ret.append(np.array(triu_dist[lptr:(lptr + rlen)]))
        lptr += rlen
    return ret


def thresh_cluster(triu_dist, dist_thresh, nr):
    triu_dist[:] = group_triu_dist(triu_dist, nr)
    par = -np.ones(nr, dtype=np.int32)
    for i in range(1, nr):  # over columns of distance matrix
        head = np.argmin([triu_dist[j][i - j - 1] for j in range(i)])
        if triu_dist[head][i - head - 1] <= dist_thresh:
            par[i] = head
    return np.where(par < 0, np.arange(nr), par)


def filter_write_clusters(labels, rules, rstrs, rstr_i, rstr2orig, args):
    clusters = {}
    for k, v in enumerate(labels):
        clusters.setdefault(v, []).append(k)
    cov = 0.
    nd = sum(rules.values())
    ri_uniq = []
    thresh_cnt = floor(nd * args.min_rule_prop)
    for p, clst in clusters.items():
        cur_cov = sum([rules[rstrs[c]] for c in clst])
        if cur_cov >= thresh_cnt:
            cov += cur_cov
            labels[clst] = len(ri_uniq)
            ri_uniq.append(p)
        else:
            clusters[p] = None
            labels[clst] = -1
    cov /= nd
    rstrs[:] = [rstr2orig[rstrs[p]] for p in ri_uniq]
    for i, l in enumerate(labels):
        rstr_i[rstr_i == i] = l
    print(f'Reduced to {len(rstrs)} rules with {cov:.4f} coverage')
    write_lst(concat_path(args, f'rule_ids_{args.cluster_method}.txt'), rstr_i)
    write_lst(concat_path(args, f'rule_{args.cluster_method}.txt'), rstrs)


def main(args):
    phrs, ctxs, tgts, phrs_orig, cpts_tgt, phr_tgt_sps = read_fs(args)
    ems = partial(fmatch_single, match_fn=exact_match, tmode='bup')
    sri = partial(SlottedRule, max_sp_width=args.max_sp_width, ignore_trail_punct=args.ignore_trail_punct)
    with Pool(args.n_proc) as p:
        res = p.map(ems, phrs, ctxs, cpts_tgt, phr_tgt_sps)
        srs = p.map(sri, phrs, phrs_orig, [get_slot_spans(r) for r in res])

    rules, rstr_i, rstr2orig, nr = count_rules(srs)
    rstrs = [t[0] for t in rules.most_common()]  # sort by decreasing frequency
    triu_dist = get_triu_dist(rstrs, nr)
    if args.cluster_method == 'thresh':
        dist_thresh = np.percentile(triu_dist, args.perc_q)
        labels = thresh_cluster(triu_dist, dist_thresh, nr)
    elif args.cluster_method == 'affinity':
        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html
        dist = triu_to_full(triu_dist, nr)
        af = AffinityPropagation(affinity='precomputed', random_state=None).fit(-dist)
        labels = af.labels_
    elif args.cluster_method == 'hierarch':
        # https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
        Z = linkage(triu_dist, method='ward')
        dist_thresh = np.percentile(triu_dist, args.perc_q)
        labels = fcluster(Z, t=dist_thresh, criterion='distance')

    filter_write_clusters(labels, rules, rstrs, rstr_i, rstr2orig, args)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train', choices=['train', 'dev', 'test'])
    ap.add_argument('--data_dir', default='canard')
    ap.add_argument('--n_proc', type=int, default=min(4, os.cpu_count()))
    ap.add_argument('--max_sp_width', type=int, default=3)
    ap.add_argument('--ignore_trail_punct', type=int, default=1)
    ap.add_argument('--min_rule_prop', type=float, default=.002)
    ap.add_argument('--perc_q', type=int, default=10, help='For --cluster_method=thresh|hierarch: percentile of pairwise distance to bound inter-cluster distance')
    ap.add_argument('--cluster_method', default='thresh', choices=['affinity', 'thresh', 'hierarch'])
    args = ap.parse_args()
    main(args)
