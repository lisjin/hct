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
from utils import fromstring, write_lst, read_lst, concat_path, compute_bleu
from utils_data import read_label_map, yield_sources_and_targets


def get_phrs_add(args):
    with Pool(args.n_proc) as p:
        with open(concat_path(args, 'cpts_uniq.txt'), encoding='utf8') as f:
            cpts_uniq = p.map(fromstring, [l.rstrip() for l in f])
    with open(concat_path(args, 'cpt_ids.txt'), encoding='utf8') as f:
        cids = [list(map(int, l.rstrip().split(','))) for l in f]
    with open(concat_path(args, args.ctx_sps_f.format(args.cluster_method))) as f:
        ctx_sps = json.load(f)
    rules = read_lst(os.path.join(args.data_dir, 'train', args.rule_f.format(args.cluster_method)))

    with open(concat_path(args, 'unfound_phrs.json'), encoding='utf8') as f:
        unfound_phrs = json.load(f)
    rstr_i, ctx_sps = zip(*[(int(x[0]), x[1:]) if x else (-1, None) for x in ctx_sps])
    j = 0
    phrs_add = {}
    rules_add = {}
    for i, (k, v) in enumerate(unfound_phrs.items()):
        k = int(k)
        phrs_add[k] = []
        rules_add[k] = []
        for v2 in v:
            ctx_leaves = list(chain.from_iterable((cpts_uniq[c].leaves() + ['[SEP]'] for c in cids[i])))[:-1]
            phrs_add[k].append([])
            rules_add[k].append(-1)
            if rstr_i[j] > -1:
                rules_add[k][-1] = rstr_i[j]
                sub_phrs = [' '.join(ctx_leaves[t[0]:t[1]]) for t in ctx_sps[j]]
                phrs_add[k][-1].extend(sub_phrs)
            j += 1
    return phrs_add, rules_add, rules


def proc_examples(args):
    if args.write:
        phrs_add, rules_add, rules = get_phrs_add(args)
        snts = []
        tgts = []
    else:
        read_dct = {}

    label_map = read_label_map(os.path.join(args.data_out_dir, 'label_map.txt'))
    converter = tagging_converter.TaggingConverter(
            tagging_converter.get_phrase_vocabulary_from_label_map(label_map))
    builder = bert_example.BertExampleBuilder(label_map, args.vocab_f, args.do_lower_case, converter, rules=rules if args.write else None, mask=args.mask)
    is_train = args.split == 'train'
    num_converted = 0
    tags, sens, cnv_ids = [], [], []
    for i, (sources, target) in enumerate(yield_sources_and_targets(
        os.path.join(args.data_dir, f'{args.split}.tsv'))):
        example, ret = builder.build_bert_example(
                sources, target,
                use_arbitrary_target_ids_for_infeasible_examples=not is_train,
                phrs_new=phrs_add.get(i, []) if args.write else None,
                rules_new=rules_add.get(i, []) if args.write else None,
                all_phr=args.all_phr)
        if args.write:  # rewritten source sentence
            snts.append(ret)
            tgts.append(target)
        elif ret:  # unfound_phrs
            read_dct[i] = ret
        if is_train and (example is None or not example.features["can_convert"]):
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
        compute_bleu(refs=tgts, hyps=snts)
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
    ap.add_argument('--ctx_sps_f', default='rule_sps_{}.json')
    ap.add_argument('--rule_f', default='rule_{}.txt')
    ap.add_argument('--hyp_f', default='snts_{}_{}.txt')
    ap.add_argument('--vocab_f', default='uncased_L-12_H-768_A-12/vocab.txt')
    ap.add_argument('--n_proc', type=int, default=min(4, os.cpu_count()))
    ap.add_argument('--do_lower_case', default=False)
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--all_phr', action='store_true')
    ap.add_argument('--max_sp_width', type=int, default=3)
    ap.add_argument('--cluster_method', default='affinity', choices=['affinity', 'thresh', 'hierarch'])
    ap.add_argument('--mask', default='_')
    args = ap.parse_args()
    main(args)
