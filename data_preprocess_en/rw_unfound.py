#!/usr/bin/env python3
import argparse
import json
import sys

import bert_example_out as bert_example
import tagging_converter

from itertools import chain
from os import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool

from tagging import EditingTask
from utils import fromstring, Metrics
from utils_data import read_label_map, yield_sources_and_targets


def get_phrs_add(args):
    with Pool(args.n_proc) as p:
        with open(args.cpts_uniq_f, encoding='utf8') as f:
            cpts_uniq = p.map(fromstring, [l.rstrip() for l in f])
    with open(args.cids_f, encoding='utf8') as f:
        cids = [list(map(int, l.rstrip().split(','))) for l in f]
    with open(args.ctx_sps_f.format(args.mmode, args.tmode)) as f:
        ctx_sps = json.load(f)
    with open(args.unmatch_path, encoding='utf8') as f:
        unfound_phrs = json.load(f)

    j = 0
    phrs_add = {}
    for i, (k, v) in enumerate(unfound_phrs.items()):
        k = int(k)
        phrs_add[k] = []
        for v2 in v['phr']:
            ctx_leaves = list(chain.from_iterable((cpts_uniq[c].leaves() + ['[SEP]'] for c in cids[i])))[:-1]
            phrs_add[k].append(' '.join(list(chain.from_iterable((ctx_leaves[t[0]:t[1]] for t in ctx_sps[j])))))
            j += 1
    return phrs_add


def proc_examples(args):
    if args.write:
        phrs_add = get_phrs_add(args)
        snts = []
        tgts = []
    else:
        unfound_dct = {}

    label_map = read_label_map(args.label_map_f)
    converter = tagging_converter.TaggingConverter(
            tagging_converter.get_phrase_vocabulary_from_label_map(label_map))
    builder = bert_example.BertExampleBuilder(label_map, args.vocab_f, args.max_seq_length, args.do_lower_case, converter)
    num_converted = 0
    for i, (sources, target) in enumerate(yield_sources_and_targets(
        args.train_f, args.train_fmt)):
        example, ret = builder.build_bert_example(
                sources, target,
                phrs_new=phrs_add.get(i, []) if args.write else None)
        if args.write:  # rewritten source sentence
            snts.append(ret)
            tgts.append(target)
        elif ret:  # unfound_phrs
            unfound_dct[i] = {'src': sources[0], 'tgt': target, 'phr': ret[:]}
        if example is None or example.features["can_convert"]==False:
            continue
        num_converted += 1

    if args.write:
        with open(args.out_f.format(args.mmode, args.tmode), 'w', encoding='utf8') as f:
            f.writelines(f'{l}\n' for l in snts)
        compute_bleu(args, tgts, snts)
    else:
        with open(args.unmatch_path, 'w', encoding='utf8') as f:
            json.dump(unfound_dct, f)


def compute_bleu(args, refs=None, hyps=None):
    if refs is None:
        refs = [target for _, target in yield_sources_and_targets(args.train_f, args.train_fmt)]
    if hyps is None:
        with open(args.out_f, encoding='utf8') as f:
            hyps = [l.rstrip() for l in f]
    cov = 0.
    for i, ref in enumerate(refs):
        if hyps[i] == ref:
            cov += 1
    print(f'EM: {cov / len(refs)}')
    bleu_tup = Metrics.bleu_score(refs, hyps)


def main(args):
    proc_examples(args)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cpts_uniq_f', default='canard/cpts_uniq.txt')
    ap.add_argument('--cids_f', default='canard/cpt_ids.txt')
    ap.add_argument('--mmode', default='difflib')
    ap.add_argument('--tmode', default='bup')
    ap.add_argument('--ctx_sps_f', default='canard/sps_{}_{}.json')
    ap.add_argument('--out_f', default='canard/snts_{}_{}.txt')
    ap.add_argument('--unmatch_path', default='canard/unfound_phrs.json')
    ap.add_argument('--label_map_f', default='canard_out/label_map.txt')
    ap.add_argument('--max_seq_length', type=int, default=128)
    ap.add_argument('--do_lower_case', default=False)
    ap.add_argument('--vocab_f', default='uncased_L-12_H-768_A-12/vocab.txt')
    ap.add_argument('--train_f', default='canard/train.tsv')
    ap.add_argument('--train_fmt', default='wikisplit')
    ap.add_argument('--n_proc', type=int, default=2 * cpu_count() // 3)
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()
    main(args)
