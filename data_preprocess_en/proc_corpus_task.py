#!/usr/bin/env python3
import argparse
import csv
import json
import os

from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial


def load_data(args, train_prop=0.8):
    def extract_dial(data):
        out = []
        for d in data:
            hist = ['~']
            for turn in d['dial']:
                rewrite = turn['usr']['transcript_complete']
                qs = []
                if turn['usr']['transcript_with_ellipsis']:
                    qs.append(turn['usr']['transcript_with_ellipsis'])
                elif turn['usr']['transcript_with_coreference']:
                    qs.append(turn['usr']['transcript_with_coreference'])
                if not qs:
                    qs.append(rewrite)
                for q in qs:
                    out.append({
                        'History': hist[-2:],
                        'Question': q,
                        'Rewrite': rewrite
                        })
                hist.append(rewrite)
                hist.append(turn['sys']['sent'])
        return out

    with open(os.path.join(args.inp_dir, f'CamRest676_annotated.json'), 'r', encoding='utf8') as f:
        data = json.load(f)
        train_end = int(train_prop * len(data))
        dev_end = train_end + (len(data) - train_end) // 2
        data_dct = {}
        data_dct['train'] = extract_dial(data[:train_end])
        data_dct['dev'] = extract_dial(data[train_end:dev_end])
        data_dct['test'] = extract_dial(data[dev_end:])
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


def get_split(get_new_sen, data, split, n_proc, keys=('History', 'Question', 'Rewrite'), st=0):
    """Process all data points based on custom `get_new_sen` function."""
    with Pool(n_proc) as p:
        return p.map(get_new_sen, [tuple(d[k] for k in keys[st:]) for d in data[split]])


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

    def proc_split(data_dct, split):
        data = get_split(get_new_sen, data_dct, split, args.n_proc)
        data_t, data_p = list(zip(*data))

        with open(os.path.join(args.inp_dir, f'{split}.tsv'), 'w', encoding='utf8', newline='') as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            tsv_writer.writerows(data_t)
        with open(os.path.join(args.inp_dir, f'{split}_pos.tsv'), 'w', encoding='utf8', newline='') as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            tsv_writer.writerows(data_p)

    data_dct = load_data(args)
    for split in args.splits:
        proc_split(data_dct, split)


def wo_context(args, std_sen):
    def get_new_sen(tup):
        inp_sen, tgt_sen = tup
        return [std_sen(inp_sen)[0], std_sen(tgt_sen)[0]]

    def proc_split(data_dct, split):
        return get_split(get_new_sen, data_dct, split, args.n_proc, st=1)

    datum = []
    data_dct = load_data(args)
    for split in args.splits:
        datum.extend(proc_split(data_dct, split))

    with open(f'{args.inp_dir}/train_valid_test_wo_context.tsv', 'w', encoding='utf8', newline='') as out_f:
        tsv_writer = csv.writer(out_f, delimiter='\t')
        tsv_writer.writerows(datum)


def main(args):
    tokenizer = SpacyTokenizer(language='en_core_web_sm')
    std_sen_ = partial(std_sen, tokenizer=tokenizer)
    if args.use_context:
        with_context(args, std_sen_)
    else:
        wo_context(args, std_sen_)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--use_context', action='store_true')
    ap.add_argument('--inp_dir', default='task')
    ap.add_argument('--splits', default=('train', 'dev', 'test'))
    ap.add_argument('--n_proc', type=int, default=min(4, os.cpu_count()))
    args = ap.parse_args()
    main(args)
