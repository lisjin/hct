#!/usr/bin/env python3
import argparse
import csv
import json
import os


def data_iterator(inp_dir, split):
    with open(os.path.join(inp_dir, f'{split}.json'), 'r', encoding='utf8') as f:
        for d in json.load(f):
            yield d


def std_sen(s):
    return s.rstrip(' *').lower()


def get_split(get_new_sen, inp_dir, split, keys=('History', 'Question', 'Rewrite'), st=0):
    """Process all data points based on custom `get_new_sen` function."""
    return [get_new_sen(*tuple(d[k] for k in keys[st:])) for d in data_iterator(inp_dir, split)]


def with_context(args):
    def get_new_sen(sens, inp_sen, tgt_sen):
        sens = [std_sen(s) for s in sens]
        return [' [SEP] '.join(sens) + ' [CI] ' + std_sen(inp_sen), std_sen(tgt_sen)]

    def proc_split(inp_dir, split):
        data = get_split(get_new_sen, inp_dir, split)
        with open(f'{inp_dir}/{split}.tsv', 'w', encoding='utf8', newline='') as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            tsv_writer.writerows(data)

    for split in args.splits:
        proc_split(args.inp_dir, split)


def wo_context(args):
    def get_new_sen(inp_sen, tgt_sen):
        return [std_sen(inp_sen), std_sen(tgt_sen)]

    def proc_split(inp_dir, split):
        return get_split(get_new_sen, inp_dir, split, st=1)

    datum = []
    for split in args.splits[:2]:
        datum.extend(proc_split(args.inp_dir, split))

    with open(f'{args.inp_dir}/train_valid_test_wo_context.tsv', 'w', encoding='utf8', newline='') as out_f:
        tsv_writer = csv.writer(out_f, delimiter='\t')
        tsv_writer.writerows(datum)


def main(args):
    if args.use_context:
        with_context(args)
    else:
        wo_context(args)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--use_context', action='store_true')
    ap.add_argument('--inp_dir', default='canard')
    ap.add_argument('--splits', default=('train', 'dev', 'test'))
    args = ap.parse_args()
    main(args)
