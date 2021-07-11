#!/usr/bin/env python3
import argparse
import json
import os
import sys

import bert_example_out as bert_example
import tagging_converter

from itertools import chain
from pathos.multiprocessing import ProcessingPool as Pool

from tagging import EditingTask
from utils import fromstring, merge_sps, Metrics, write_lst, concat_path
from utils_data import read_label_map, yield_sources_and_targets


def get_phrs_add(args):
    with Pool(args.n_proc) as p:
        with open(concat_path(args, 'cpts_uniq.txt'), encoding='utf8') as f:
            cpts_uniq = p.map(fromstring, [l.rstrip() for l in f])
    with open(concat_path(args, 'cpt_ids.txt'), encoding='utf8') as f:
        cids = [list(map(int, l.rstrip().split(','))) for l in f]
    with open(concat_path(args, args.ctx_sps_f.format(args.mmode, args.tmode))) as f:
        ctx_sps = json.load(f)
    with open(concat_path(args, 'unfound_phrs.json'), encoding='utf8') as f:
        unfound_phrs = json.load(f)

    j = 0
    phrs_add = {}
    for i, (k, v) in enumerate(unfound_phrs.items()):
        k = int(k)
        phrs_add[k] = []
        for v2 in v:
            ctx_leaves = list(chain.from_iterable((cpts_uniq[c].leaves() + ['[SEP]'] for c in cids[i])))[:-1]
            phrs_add[k].append([])
            for t in merge_sps(ctx_sps[j]):
                if t[1] - t[0]:
                    cand_phr = ' '.join(ctx_leaves[t[0]:t[1]])
                    if cand_phr != '[SEP]':
                        phrs_add[k][-1].append(cand_phr)
            j += 1
    return phrs_add


def compute_bleu(hyp_path, refs=None, hyps=None):
    if refs is None:
        refs = [target for _, target in yield_sources_and_targets(os.path.join(args.data_dir, f'{args.split}.tsv'), args.tsv_fmt)]
    if hyps is None:
        with open(hyp_path, encoding='utf8') as f:
            hyps = [l.rstrip() for l in f]
    cov = 0.
    for i, ref in enumerate(refs):
        if hyps[i] == ref:
            cov += 1
    print(f'EM: {cov / len(refs)}')
    bleu_tup = Metrics.bleu_score(refs, hyps)


def proc_examples(args):
    if args.write:
        phrs_add = get_phrs_add(args)
        snts = []
        tgts = []
    else:
        read_dct = {}

    label_map = read_label_map(os.path.join(args.data_out_dir, 'label_map.txt'))
    converter = tagging_converter.TaggingConverter(
            tagging_converter.get_phrase_vocabulary_from_label_map(label_map))
    builder = bert_example.BertExampleBuilder(label_map, args.vocab_f, args.max_seq_length, args.do_lower_case, converter)
    is_train = args.split == 'train'
    num_converted = 0
    tags, sens, cnv_ids = [], [], []
    for i, (sources, target) in enumerate(yield_sources_and_targets(
        os.path.join(args.data_dir, f'{args.split}.tsv'), args.tsv_fmt)):
        example, ret = builder.build_bert_example(
                sources, target,
                phrs_new=phrs_add.get(i, []) if args.write else None,
                all_phr=args.all_phr)
        if args.write:  # rewritten source sentence
            snts.append(ret)
            tgts.append(target)
        elif ret:  # unfound_phrs
            read_dct[i] = ret
        if is_train and (example is None or example.features["can_convert"]==False):
            continue
        elif is_train:
            cnv_ids.append(i)
        tags.append(' '.join([str(s) for s in example.features['labels']]))
        sens.append(' '.join(example.features['input_tokens']).replace('[CI]', '|'))
        num_converted += 1

    if args.write:
        hyp_path = concat_path(args, args.hyp_f.format(args.mmode, args.tmode))
        write_lst(hyp_path, snts)
        write_lst(concat_path(args, 'tags.txt', data_out=True), tags)
        write_lst(concat_path(args, 'sentences.txt', data_out=True), sens)
        with open(concat_path(args, 'cnv_ids.json'), 'w') as f:
            json.dump(cnv_ids, f)
        compute_bleu(hyp_path, tgts, snts)
    else:
        pref = 'all' if args.all_phr else 'unfound'
        with open(concat_path(args, f'{pref}_phrs.json'), 'w', encoding='utf8') as f:
            json.dump(read_dct, f)


def main(args):
    proc_examples(args)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train', choices=['train', 'dev', 'test'])
    ap.add_argument('--data_dir', default='canard')
    ap.add_argument('--data_out_dir', default='canard_out')
    ap.add_argument('--mmode', default='difflib')
    ap.add_argument('--tmode', default='bup')
    ap.add_argument('--ctx_sps_f', default='sps_{}_{}.json')
    ap.add_argument('--hyp_f', default='snts_{}_{}.txt')
    ap.add_argument('--vocab_f', default='uncased_L-12_H-768_A-12/vocab.txt')
    ap.add_argument('--tsv_fmt', default='wikisplit')
    ap.add_argument('--n_proc', type=int, default=min(4, os.cpu_count()))
    ap.add_argument('--max_seq_length', type=int, default=128)
    ap.add_argument('--do_lower_case', default=False)
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--all_phr', action='store_true')
    args = ap.parse_args()
    main(args)
