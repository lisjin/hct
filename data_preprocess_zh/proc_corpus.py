#!/usr/bin/env python3
import argparse
import csv
import jieba
import os


def is_all_chinese(word):
    # identify whether all chinese characters
    for _char in word:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def cut_mixed_sentence(text):
    # for chinese, return character; for english, return word;
    jieba_words = list(jieba.cut(text))
    ret_chars = []
    for word in jieba_words:
        if is_all_chinese(word):
            ret_chars.extend(list(word))
        else:
            ret_chars.append(word)
    return ' '.join(ret_chars)


def with_context(args, total_lines, total_len, train_prop=0.9):
    train_end = int(train_prop * total_len)
    eval_len = (total_len - train_end) // 2
    data_dct = {}
    data_dct['train'] = total_lines[:train_end]
    data_dct['dev'] = total_lines[train_end:(train_end + eval_len)]
    data_dct['test'] = total_lines[(train_end + eval_len):]

    def get_new_sen(sentences):
        new_sen = [cut_mixed_sentence(s) for s in sentences]
        return [' [SEP] '.join(new_sen[:-2]) + ' [CI] ' + new_sen[-2], new_sen[-1]]

    def load_write(_data, split):
        for _ind in range(len(_data)):
            sentences = _data[_ind].split('\t\t')
            _data[_ind] = get_new_sen(sentences)
        with open(os.path.join(args.inp_dir, f'{split}.tsv'), 'w', encoding='utf8', newline='') as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            tsv_writer.writerows(_data)

    for split in args.splits:
        load_write(data_dct[split], split)


def wo_context(args, total_lines, total_len):
    for i, line in enumerate(total_lines):
        total_lines[i] = [cut_mixed_sentence(l) for l in line.split('\t\t')[-2:]]

    with open(os.path.join(args.inp_dir, 'train_valid_test_wo_context.tsv'), 'w', encoding='utf8', newline='') as out_f:
        tsv_writer = csv.writer(out_f, delimiter='\t')
        tsv_writer.writerows(total_lines)


def main(args):
    with open(os.path.join(args.inp_dir, 'corpus.txt'), "r", encoding="utf8") as f:
        total_lines = [line.strip() for line in f.readlines()]
        total_len = len(total_lines)

        if args.use_context:
            with_context(args, total_lines, total_len)
        else:
            wo_context(args, total_lines, total_len)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--use_context', action='store_true')
    ap.add_argument('--inp_dir', default='rewrite')
    ap.add_argument('--splits', default=('train', 'dev', 'test'))
    args = ap.parse_args()
    main(args)
