#!/usr/bin/env python3
import json
import sys

from functools import partial
from itertools import chain
from nltk import Tree


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def _read_leaf(x):
    return x.replace('{', '(').replace('}', ')')


fromstring = partial(Tree.fromstring, read_leaf=_read_leaf)


def expand_flat(lst, phr_tgt_sps):
    return list(chain.from_iterable((lst[i] for _ in l2) for i, l2 in enumerate(
        phr_tgt_sps)))


def expand_phrs(tgts, phr_tgt_sps):
    return [tgts[i].split()[t[0]:t[0]+t[1]] for i, l2 in enumerate(phr_tgt_sps)\
            for t in l2]


def read_lem(lem_f):
    with open(lem_f, encoding='utf8') as f:
        lems = json.load(f)
        ctxs, tgts = lems[::2], lems[1::2]
    return ctxs, tgts


def read_expand(phr_tgt_sps_f, ctxs=None, tgts=None, cpts_tgt=None):
    """Use nested list structure of phr_tgt_sps to expand other data lists."""
    with open(phr_tgt_sps_f, encoding='utf8') as f:
        phr_tgt_sps = json.load(f)
    if ctxs:
        ctxs[:] = expand_flat(ctxs, phr_tgt_sps)
    phrs = None
    if tgts:
        tgts[:] = expand_flat(tgts, phr_tgt_sps)
        phrs = expand_phrs(tgts, phr_tgt_sps)
    if cpts_tgt:
        cpts_tgt = expand_flat(cpts_tgt, phr_tgt_sps)
    return cpts_tgt, ctxs, tgts, phrs, phr_tgt_sps


def tightest_span(t, i, k, pl):
    if i > 0 or i + k < pl:
        return None
    sp_b = None
    i2 = i
    for st in t:
        if type(st) is Tree:
            k2 = len(st.leaves())
            o = tightest_span(st, i2, k2, pl)
            if o is not None:
                sp_b = o
                break
            i2 += k2
        else:
            i2 += 1
    return sp_b if sp_b else (t, i, k)
