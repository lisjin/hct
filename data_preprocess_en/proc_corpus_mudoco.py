#!/usr/bin/env python3
import argparse
import csv
import json
import os

from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial


def load_data(args):
    def extract_dial(data, data_dct):
        # Adapted from https://github.com/apple/ml-cread/blob/main/modeling/utils/process_data.py
        for dial in data['dialogs'].values():
            hist = []
            split = dial['split'] if dial['split'] != 'eval' else 'dev'
            for turn_i, turn in enumerate(dial['turns']):
                if turn_i + 1 != turn['number'] or not turn['utterance']:
                    break
                if turn_i % 2 == 0:  # user
                    data_dct[split].append({
                            'history': hist[:] if len(hist) else ['~'],
                            'utterance': turn['utterance'],
                            'rewrite': turn['rewritten_utterance'] if\
                                    turn['graded'] else turn['utterance']
                            })
                hist.append(turn['utterance'])
    data_dct = {k: [] for k in args.splits}
    domain_rng = {k: {} for k in args.splits}
    for domain in args.domains:
        for split in args.splits:
            domain_rng[split]['sz'] = len(data_dct[split])
        with open(os.path.join(args.inp_dir, f'mudoco_{domain}.json'), 'r', encoding='utf8') as f:
            extract_dial(json.load(f), data_dct)
        if domain == 'calling':
            domain_rng['train'][domain] = (domain_rng['train']['sz'], len(data_dct['train']))
        if domain in args.domains:
            for split in ('dev', 'test'):
                domain_rng[split][domain] = (domain_rng[split]['sz'], len(data_dct[split]))
    for split in args.splits:
        s_len = len(data_dct[split])
        del domain_rng[split]['sz']

    domain_rng_path = os.path.join(args.inp_dir, 'domain_rng.json')
    if not os.path.isfile(domain_rng_path):
        with open(domain_rng_path, 'w', encoding='utf8') as f:
            json.dump(domain_rng, f)
    return data_dct


def std_sen(s, tokenizer):
    s = s.replace('"', '')
    toks, pos = tokenizer.tokenize(s), ''
    try:
        toks, pos = list(zip(*[(t.text.lower(), t.tag_) for t in toks]))
        pos = ' '.join(pos)
    except ValueError:
        toks = []
    return ' '.join(toks), pos


def get_split(get_new_sen, data, split, n_proc, keys=('history', 'utterance', 'rewrite'), st=0):
    """Process all data points based on custom `get_new_sen` function."""
    with Pool(n_proc) as p:
        return p.map(get_new_sen, [tuple(d[k] for k in keys[st:]) for d in data[split]])


def get_tok_suf(tokenize, use_pos=False):
    return '_tok' if tokenize else ('_pos' if use_pos else '')


def with_context(args, std_sen, data_dct):
    def get_new_sen(tup):
        sens, inp_sen, tgt_sen = tup
        sens, sens_p = list(zip(*[std_sen(s) for s in sens]))
        inp_t, inp_p = std_sen(inp_sen)
        tgt_t, tgt_p = std_sen(tgt_sen)
        toks = [' [SEP] '.join(sens) + ' [CI] ' + inp_t, tgt_t]
        pos = [' [SEP] '.join(sens_p) + ' [CI] ' + inp_p, tgt_p]
        return toks, pos

    def proc_split(data_dct, split):
        data = get_split(get_new_sen, data_dct, split, args.n_proc)
        data_t, data_p = list(zip(*data))

        with open(os.path.join(args.inp_dir, f'{split}.tsv'), 'w', encoding='utf8', newline='') as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            tsv_writer.writerows(data_t)
        with open(os.path.join(args.inp_dir, f'{split}_pos.tsv'), 'w', encoding='utf8', newline='') as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            tsv_writer.writerows(data_p)

    for split in args.splits:
        proc_split(data_dct, split)


def wo_context(args, std_sen, data_dct):
    def get_new_sen(tup):
        inp_sen, tgt_sen = tup
        return [std_sen(inp_sen)[0], std_sen(tgt_sen)[0]]

    def proc_split(data_dct, split):
        return get_split(get_new_sen, data_dct, split, args.n_proc, st=1)

    datum = []
    for split in args.splits:
        datum.extend(proc_split(data_dct, split))

    with open(f'{args.inp_dir}/train_valid_test_wo_context.tsv', 'w', encoding='utf8', newline='') as out_f:
        tsv_writer = csv.writer(out_f, delimiter='\t')
        tsv_writer.writerows(datum)


def main(args):
    tokenizer = SpacyTokenizer(language='en_core_web_sm')
    std_sen_ = partial(std_sen, tokenizer=tokenizer)
    data_dct = load_data(args)
    if args.with_context:
        with_context(args, std_sen_, data_dct)
    if args.wo_context:
        wo_context(args, std_sen_, data_dct)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--with_context', action='store_true')
    ap.add_argument('--wo_context', action='store_true')
    ap.add_argument('--inp_dir', default='mudoco')
    ap.add_argument('--splits', default=('train', 'dev', 'test'))
    ap.add_argument('--domains', default=('calling', 'messaging', 'music', 'news', 'reminders', 'weather'))
    ap.add_argument('--n_proc', type=int, default=min(4, os.cpu_count()))
    args = ap.parse_args()
    main(args)
