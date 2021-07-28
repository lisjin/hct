#!/usr/bin/env python3
import json
import os
import sys

from functools import partial
from itertools import chain
from nltk import Tree

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from score import Metrics
from utils_data import yield_sources_and_targets


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def _read_leaf(x):
    return x.replace('{', '(').replace('}', ')')


fromstring = partial(lambda x, rl: Tree.fromstring(x, read_leaf=rl) if x else Tree(None, []), rl=_read_leaf)


def expand_flat(lst, phr_tgt_sps):
    return list(chain.from_iterable((lst[i] for _ in l2) for i, l2 in enumerate(
        phr_tgt_sps)))


def expand_phrs(tgts, phr_tgt_sps):
    return [tgts[i].split()[t[0]:t[0]+t[1]] for i, l2 in enumerate(phr_tgt_sps)\
            for t in l2]


def concat_path(args, fname, data_out=False):
    return os.path.join(args.data_dir if not data_out else args.data_out_dir,
            args.split, fname)


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
        phrs = expand_phrs(tgts, phr_tgt_sps)
        tgts[:] = expand_flat(tgts, phr_tgt_sps)
    if cpts_tgt:
        cpts_tgt = expand_flat(cpts_tgt, phr_tgt_sps)
    return cpts_tgt, ctxs, tgts, phrs, phr_tgt_sps


def read_lst(f_name):
    with open(f_name, 'r', encoding='utf8') as f:
        return [l.strip() for l in f]


def write_lst(f_name, lst):
    with open(f_name, 'w', encoding='utf8') as f:
        f.writelines(f'{l}\n' for l in lst)


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


def merge_sps(lst):
    if len(lst) > 1:
        par_i = -1
        for j, sp in enumerate(lst[1:]):
            if lst[j][1] == sp[0]:
                lst[j + 1][0] = -1  # mark for deletion
                if par_i == -1:
                    par_i = j
                if j == len(lst[1:]) - 1:  # write last element to parent
                    lst[par_i][1] = sp[1]
            elif par_i > -1:
                lst[par_i][1] = lst[j][1]
                par_i = -1
        lst[:] = [sp for sp in lst if sp[0] > -1]
    return lst


def ilst2str(lst):
    return ','.join([str(s) for s in lst])


def find_subspans(t, i, k, en, li, sps):
    if i == li and i + k <= en:
        sps.append((i, k))
        li += k
    elif i + k > li:
        i2 = i
        for st in t:
            k2 = len(st.leaves()) if type(st) is Tree else 1
            li = find_subspans(st, i2, k2, en, li, sps)
            i2 += k2
            if i2 >= en or li >= en:
                break
    return li


def align_phr_tgt(phr_lst, target):
    phr_spls = [phr.split() for phr in phr_lst]
    tgt_spl = target.split()
    sps = []
    for ps in phr_spls:
        pl = len(ps)
        sp = (-1, -1)
        for i in range(len(tgt_spl)):
            if tgt_spl[i:i+pl] == ps:
                sp = (i, pl)
                break
        sps.append(sp)
    return sps


def compute_bleu(refs=None, hyps=None, args=None, hyp_path=None):
    if refs is None:
        refs = [target for _, target in yield_sources_and_targets(
            os.path.join(args.data_dir, f'{args.split}.tsv'), args.tsv_fmt)]
    if hyps is None:
        with open(hyp_path, encoding='utf8') as f:
            hyps = [l.rstrip() for l in f]
    cov = 0.
    for i, ref in enumerate(refs):
        if hyps[i] == ref:
            cov += 1
    print(f'EM: {cov / len(refs)}')
    bleu_tup = Metrics.bleu_score(refs, hyps)
