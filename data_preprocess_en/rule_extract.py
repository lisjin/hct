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
    def __init__(self, tokens, orig_tokens, slot_spans, ctx_spans, max_sp_width, ignore_trail_punct, mask, validate=False):
        self.tokens = tokens
        self.orig_tokens = orig_tokens
        self.maybe_remove_punct(ignore_trail_punct)
        self.slot_spans = [(max(0, i), min(i + k, len(self))) for i, k in slot_spans]
        self.ctx_spans = ctx_spans
        self.n_slots = len(self.slot_spans)
        self.term_spans, self.n_term_spans, self.n_terms = self.get_term(slot_spans)
        if self.n_slots > max_sp_width:
            self.bound_spans(max_sp_width)
        self.max_sp_width = max_sp_width
        self.mask = mask
        if validate:
            self.validate()

    def __len__(self):
        return len(self.tokens)

    def __str__(self, use_lemma=True):
        if len(self.term_spans):
            tptr, sptr = 0, 0
            tmax, smax = len(self.term_spans), len(self.slot_spans)
            rule_str = []
            term_tokens = self.term_tokens if use_lemma else self.term_orig_tokens
            while tptr < tmax and sptr < smax:
                if self.slot_spans[sptr][0] < self.term_spans[tptr][0]:
                    rule_str.append(self.mask)
                    sptr += 1
                    while sptr < tmax and self.term_spans[sptr - 1][1] ==\
                            self.term_spans[sptr][0]:
                        rule_str.append(self.mask)
                        sptr += 1
                else:
                    rule_str.append(' '.join(term_tokens[tptr]))
                    tptr += 1
            if sptr < smax:
                rule_str.extend([self.mask] * (smax - sptr))
            elif tptr < tmax:
                rule_str.extend([' '.join(x) for x in self.term_tokens[tptr:]])
            rule_str = ' '.join(rule_str)
            return rule_str
        return ' '.join([self.mask] * min(len(self.slot_spans), self.max_sp_width))

    @property
    def slot_tokens(self):
        return self.span_tokens(self.slot_spans)

    @property
    def term_tokens(self):
        return self.span_tokens(self.term_spans)

    @property
    def term_orig_tokens(self):
        return [self.orig_tokens[s:e] for s, e in self.term_spans]

    def maybe_remove_punct(self, ignore_trail_punct):
        if ignore_trail_punct and self.tokens and self.tokens[-1] in punct_set:
            self.tokens[:] = self.tokens[:-1]
            self.orig_tokens[:] = self.orig_tokens[:-1]

    def bound_spans(self, max_sp_width):
        self.n_slots = max_sp_width
        self.slot_spans[:] = self.slot_spans[:self.n_slots]
        self.ctx_spans[:] = self.ctx_spans[:self.n_slots]
        last_slot_i = self.slot_spans[-1][1]
        for i, ts in enumerate(self.term_spans):
            if ts[0] == last_slot_i:
                self.term_spans[:] = self.term_spans[:(i + 1)]
                break
        self.n_terms = len(self.term_spans)
        nt = max(self.term_spans[-1][1] if self.term_spans else 0,
                self.slot_spans[-1][1] if self.slot_spans else 0)
        self.tokens[:] = self.tokens[:nt]

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

    with open(concat_path(args, 'unfound_phrs.json'), encoding='utf8') as f:
        outs_k = set([int(k) for k in json.load(f).keys()])
    tgts_orig = [target for _, _, target in filter_sources_and_targets(
        os.path.join(args.data_dir, f'{args.split}.tsv'), outs_k)]
    _, _, _, phrs_orig, _ = read_expand(phr_tgt_sps_f, tgts=tgts_orig)

    assert(len(phrs) == len(ctxs) == len(tgts) == len(phrs_orig) == len(phr_tgt_sps) == len(cpts_tgt))
    return phrs, ctxs, tgts, phrs_orig, cpts_tgt, phr_tgt_sps


def count_rules(srs):
    """Rule frequency across data instances.
    Returns:
        rules: Counter object keyed by rule strings
        rstr_i: Size len(srs) list of rule string indices in rstrs
        rstrs: List of unique rule strings, sorted by decreasing frequency
        nr: Number of unique rule strings
    """
    rules = Counter()
    rstr2orig = {}
    rstr_i = []
    for i, sr in enumerate(srs):
        rstr = str(sr)
        if rstr:
            if rstr not in rules:
                rstr2orig[rstr] = sr.__str__(use_lemma=False)
            rstr_i.append(rstr)
            rules[rstr] += 1
        else:
            rstr_i.append('')
    nr = len(rules)
    print(f'Found {nr} rules')

    rstrs = [t[0] for t in rules.most_common()]  # sort by decreasing frequency
    rstr2i = {rstr: i for i, rstr in enumerate(rstrs)}
    rstr_i = np.array([rstr2i.get(rstr, -1) for rstr in rstr_i])
    return rules, rstr_i, rstrs, rstr2orig, rstr2i, nr


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


def get_str_slots(inp_str, mask):
    return len([c for c in inp_str.split() if c == mask])


def get_mask_rep(n_mask, mask):
    return ' '.join([mask] * n_mask)


def filter_write_clusters(srs, labels, rules, rstrs, rstr_i, rstr2orig, rstr2i, args):
    clusters = {}
    nd = sum(rules.values())
    thresh_cnt = floor(nd * args.min_rule_prop)
    n_slots_dct = {}
    for k, v in enumerate(labels):
        if v not in n_slots_dct:
            n_slots_dct[v] = get_str_slots(rstrs[v], args.mask)
        k_slots = get_str_slots(rstrs[k], args.mask) if k not in n_slots_dct\
                else n_slots_dct[k]
        if k_slots != n_slots_dct[v]:
            v = k if rules[rstrs[k]] >= thresh_cnt or k_slots == 0 else\
                    rstr2i[get_mask_rep(k_slots, args.mask)]
            if v not in n_slots_dct:
                n_slots_dct[v] = k_slots
        clusters.setdefault(v, []).append(k)

    cov = 0.
    rstrs_uniq = {}
    for p, clst in clusters.items():
        cur_cov = sum([rules[rstrs[c]] for c in clst])
        if cur_cov >= thresh_cnt:
            cov += cur_cov
            rstrs_uniq[p] = len(rstrs_uniq)
            for c in clst:
                rstr_i[rstr_i == c] = rstrs_uniq[p]
        else:
            clusters[p] = None
            for c in clst:
                rstr_i[rstr_i == c] = rstrs_uniq[rstr2i[get_mask_rep(
                    n_slots_dct[p], args.mask)]] if n_slots_dct[p] > 0 else -1
    cov /= nd

    rstrs[:] = [rstr2orig[rstrs[p]] for p in rstrs_uniq.keys()]
    print(f'Reduced to {len(rstrs)} rules with {cov:.4f} coverage')

    rule_sps = [(str(ri), *cs) if ri > -1 else None for ri, cs in zip(rstr_i, [sr.ctx_spans for sr in srs])]
    with open(concat_path(args, f'rule_sps_{args.cluster_method}.json'), 'w', encoding='utf8') as f:
        json.dump(rule_sps, f)
    write_lst(concat_path(args, f'rule_{args.cluster_method}.txt'), rstrs)


def get_spans(r):
    if not r:
        return [], []
    slot_spans, ctx_spans = zip(*[(r2[:2], r2[-1]) for r2 in r])
    return slot_spans, list(ctx_spans)


def main(args):
    phrs, ctxs, tgts, phrs_orig, cpts_tgt, phr_tgt_sps = read_fs(args)
    ems = partial(fmatch_single, match_fn=exact_match, tmode='bup')
    sri = partial(SlottedRule, max_sp_width=args.max_sp_width, ignore_trail_punct=args.ignore_trail_punct, mask=args.mask)
    res = [ems(*t) for t in list(zip(phrs, ctxs, cpts_tgt, phr_tgt_sps))]
    _, sps_out, _ = zip(*res)
    srs = [sri(phr, phr_orig, *get_spans(r)) for phr, phr_orig, r in zip(phrs, phrs_orig, sps_out)]

    rules, rstr_i, rstrs, rstr2orig, rstr2i, nr = count_rules(srs)
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

    filter_write_clusters(srs, labels, rules, rstrs, rstr_i, rstr2orig, rstr2i, args)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train', choices=['train', 'dev', 'test'])
    ap.add_argument('--data_dir', default='canard')
    ap.add_argument('--n_proc', type=int, default=min(4, os.cpu_count()))
    ap.add_argument('--max_sp_width', type=int, default=3)
    ap.add_argument('--mask', default='_')
    ap.add_argument('--ignore_trail_punct', type=int, default=1)
    ap.add_argument('--min_rule_prop', type=float, default=.003)
    ap.add_argument('--perc_q', type=int, default=10, help='For --cluster_method=thresh|hierarch: pairwise distance percentile for upper-bound on inter-cluster distance')
    ap.add_argument('--cluster_method', default='affinity', choices=['affinity', 'thresh', 'hierarch'])
    args = ap.parse_args()
    main(args)
