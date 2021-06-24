#!/usr/bin/env python3
import argparse
import csv
import json
import os

from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from pathos.multiprocessing import ProcessingPool as Pool
from bert import tokenization
from functools import partial


def data_iterator(inp_dir, split):
    with open(os.path.join(inp_dir, f'{split}.json'), 'r', encoding='utf8') as f:
        for d in json.load(f):
            yield d


def std_sen(s, tokenizer):
    s = s.replace('"', '')
    toks, pos = tokenizer.tokenize(s), ''
    try:
        toks, pos = list(zip(*[(t.text.lower(), t.tag_) for t in toks]))
        pos = ' '.join(pos)
    except ValueError:
        toks = []
    return ' '.join(toks), pos


def get_split(get_new_sen, inp_dir, split, n_proc, keys=('History', 'Question', 'Rewrite'), st=0):
    """Process all data points based on custom `get_new_sen` function."""
    with Pool(n_proc) as p:
        return p.map(get_new_sen, [tuple(d[k] for k in keys[st:]) for d in data_iterator(inp_dir, split)])


def get_tok_suf(tokenize, use_pos=False):
    return '_tok' if tokenize else ('_pos' if use_pos else '')


def with_context(args, std_sen):
    def get_new_sen(tup):
        sens, inp_sen, tgt_sen = tup
        sens, sens_p = list(zip(*[std_sen(s) for s in sens]))
        inp_t, inp_p = std_sen(inp_sen)
        tgt_t, tgt_p = std_sen(tgt_sen)
        toks = [' [SEP] '.join(sens) + ' [CI] ' + inp_t, tgt_t]
        pos = [' [SEP] '.join(sens_p) + ' [CI] ' + inp_p, tgt_p]
        return toks, pos

    def proc_split(inp_dir, split):
        data = get_split(get_new_sen, inp_dir, split, args.n_proc)
        data_t, data_p = list(zip(*data))

        with open(f'{inp_dir}/{split}.tsv', 'w', encoding='utf8', newline='') as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            tsv_writer.writerows(data_t)
        with open(f'{inp_dir}/{split}_pos.tsv', 'w', encoding='utf8', newline='') as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            tsv_writer.writerows(data_p)

    for split in args.splits:
        proc_split(args.inp_dir, split)


def wo_context(args, std_sen):
    def get_new_sen(tup):
        inp_sen, tgt_sen = tup
        return [std_sen(inp_sen)[0], std_sen(tgt_sen)[0]]

    def proc_split(inp_dir, split):
        return get_split(get_new_sen, inp_dir, split, args.n_proc, st=1)

    datum = []
    for split in args.splits:
        datum.extend(proc_split(args.inp_dir, split))

    with open(f'{args.inp_dir}/train_valid_test_wo_context.tsv', 'w', encoding='utf8', newline='') as out_f:
        tsv_writer = csv.writer(out_f, delimiter='\t')
        tsv_writer.writerows(datum)


def main(args):
    tokenizer = SpacyTokenizer(language='en_core_web_sm')
    std_sen2 = partial(std_sen, tokenizer=tokenizer)
    if args.use_context:
        with_context(args, std_sen2)
    else:
        wo_context(args, std_sen2)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--use_context', action='store_true')
    ap.add_argument('--inp_dir', default='canard')
    ap.add_argument('--splits', default=('train', 'dev', 'test'))
    ap.add_argument('--n_proc', type=int, default=2 * os.cpu_count() // 3)
    args = ap.parse_args()
    main(args)
