#!/usr/bin/env python3
import argparse
import csv
import jieba


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


def with_context(total_lines, total_len):
    border = int(0.9 * total_len)
    train_data = total_lines[:border]
    dev_data = total_lines[border:]

    def get_new_sen(sentences):
        new_sen = [cut_mixed_sentence(s) for s in sentences]
        return [' [SEP] '.join(new_sen[:-2]) + ' [CI] ' + new_sen[-2], new_sen[-1]]

    for train_ind in range(len(train_data)):
        sentences = train_data[train_ind].split('\t\t')
        train_data[train_ind] = get_new_sen(sentences)

    for dev_ind in range(len(dev_data)):
        sentences = dev_data[dev_ind].split('\t\t')
        dev_data[dev_ind] = get_new_sen(sentences)

    with open("data/train.tsv", "w", encoding="utf8", newline='') as train_f:
        tsv_writer = csv.writer(train_f, delimiter='\t')
        tsv_writer.writerows(train_data)

    with open("data/dev.tsv", "w", encoding="utf8", newline='') as dev_f:
        tsv_writer = csv.writer(dev_f, delimiter='\t')
        tsv_writer.writerows(dev_data)


def wo_context(total_lines, total_len):
    for i, line in enumerate(total_lines):
        total_lines[i] = [cut_mixed_sentence(l) for l in line.split('\t\t')[-2:]]

    with open('data/train_valid_test_wo_context.tsv', 'w', encoding='utf8', newline='') as out_f:
        tsv_writer = csv.writer(out_f, delimiter='\t')
        tsv_writer.writerows(total_lines)


def main(args):
    origin_file = "data/corpus.txt"
    with open(origin_file, "r", encoding="utf8") as f:
        total_lines = [line.strip() for line in f.readlines()]
        total_len = len(total_lines)

        if args.use_context:
            with_context(total_lines, total_len)
        else:
            wo_context(total_lines, total_len)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--use_context', action='store_true')
    args = ap.parse_args()
    main(args)
