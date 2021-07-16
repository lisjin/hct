#!/usr/bin/env python3
import argparse
import json
import numpy as np

from collections import Counter
from functools import partial
from itertools import chain
from nltk import Tree
from os import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool

from fmatch_lem import fmatch_single
from utils import fromstring, read_lem, read_expand, concat_path


class SlottedRule:
    def __init__(self, tokens, slot_spans, validate=False):
        self.tokens = token
        self.slot_spans = [(max(0, i), min(i + k, len(self))) for i, k in slot_spans]
        self.n_slots = len(self.slot_spans)
        self.term_spans, self.n_term_spans, self.n_terms = self.get_term(slot_spans)
        if validate:
            self.validate()

    def __len__(self):
        return len(self.tokens)

    def __str__(self):
        rule_str = ' [MASK] '.join([' '.join(l) for l in self.term_tokens])
        if len(self.term_spans):
            pref = '[MASK] ' if self.term_spans[0][0] > 0 else ''
            suf = ' [MASK]' if self.term_spans[-1][1] < len(self) else ''
            return f'{pref}{rule_str}{suf}'
        else:
            return ' '.join(['[MASK]'] * len(self.slot_spans))

    @property
    def slot_tokens(self):
        return self.span_tokens(self.slot_spans)

    @property
    def term_tokens(self):
        return self.span_tokens(self.term_spans)

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
        return sps, len(sps), n_term

    def validate(self):
        s1 = set(chain.from_iterable(range(*sp) for sp in self.slot_spans +\
                self.term_spans))
        assert(s1 == set(range(len(self))))


def exact_match(ctx, phr):
    a = ctx.find(phr)
    if a > -1:
        return a, len(phr)
    return -1, 0


def read_fs(args):
    with Pool(args.n_proc) as p:
        with open(concat_path(args, 'cpts_tgt.txt'), encoding='utf8') as f:
            cpts_tgt = p.map(fromstring, [l.rstrip() for l in f])
    ctxs, tgts = read_lem(concat_path(args, 'unmatch_lems.json'))
    cpts_tgt, ctxs, tgts, phrs, phr_tgt_sps = read_expand(concat_path(args,
        'phr_tgt_sps.json'), ctxs=ctxs, tgts=tgts, cpts_tgt=cpts_tgt)
    phr_tgt_sps = [sp for pts in phr_tgt_sps for sp in pts]
    assert(len(phrs) == len(ctxs) == len(tgts) == len(phr_tgt_sps) == len(cpts_tgt))
    return phrs, ctxs, tgts, cpts_tgt, phr_tgt_sp


def get_slot_spans(r):
    return [r2[:2] for r2 in r[1]]


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

    rules = Counter()
    for sr in srs:
        rules[str(sr)] += 1

    print(f'Found {len(rules)} rules')
    for tup in rules.most_common(25):
        print(tup)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train', choices=['train', 'dev', 'test'])
    ap.add_argument('--data_dir', default='canard')
    ap.add_argument('--n_proc', type=int, default=min(4, cpu_count()))
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--out_f', default='rule_{}_{}.json')
    args = ap.parse_args()
    main(args)
