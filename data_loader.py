"""Data loader"""
import os
import torch
import random
import numpy as np

from pathos.multiprocessing import ProcessingPool as Pool
from transformers import BertTokenizer


class DataLoader(object):
    def __init__(self, data_dir, bert_class, params, token_pad_idx=0, tag_pad_idx=-1, rule_pad_idx=-1):
        self.data_dir = data_dir
        self.batch_size = params.batch_size
        self.max_len = params.max_len
        self.device = params.device
        self.seed = params.seed
        self.token_pad_idx = token_pad_idx
        self.tag_pad_idx = params.pad_tag_id = tag_pad_idx
        self.rule_pad_idx = rule_pad_idx
        self.max_sp_len = params.max_sp_len
        self.to_int = lambda x: int(x) + 1
        self.tokenizer = BertTokenizer.from_pretrained(bert_class)

        self.idx2tag = params.idx2tag = self.load_tags()
        self.tag2idx = params.tag2idx = {tag: idx for idx, tag in enumerate(self.idx2tag)}

    @staticmethod
    def load_tags():
        return ["KEEP", "DELETE"]

    @staticmethod
    def upd_by_ptr(seq_width, curr_len_list, rule_seq, ptr):
        rem = sw = seq_width[ptr]
        cur_len, cur_rule = curr_len_list[ptr], rule_seq[ptr]
        return rem, sw, cur_len, cur_rule, ptr + 1

    def _split_to_wordpieces_span(self, tokens, label_action, label_start, label_end, seq_width, rule_seq):
        bert_tokens = []
        bert_label_action = []
        source_indices = []
        cum_num_list = []
        curr_len_list = []
        cum_num = 0
        src_start = orig_start = len(tokens)
        for i, token in enumerate(tokens):
            pieces = self.tokenizer.tokenize(token)
            if token == '|':
                src_start = len(bert_tokens) + 1
                orig_start = i + 1

            bert_label_action.extend([label_action[i]] * len(pieces))
            bert_tokens.extend(pieces)
            curr_len_list.append(len(pieces))
            cum_num_list.append(cum_num)
            cum_num += len(pieces) - 1

        if len(bert_tokens) > self.max_len:
            new_len = self.max_len - (len(bert_tokens) - src_start)
            source_indices = list(range(new_len, self.max_len))
            bert_tokens = bert_tokens[:new_len] + bert_tokens[src_start:]
        else:
            new_len = src_start
            source_indices = list(range(src_start, len(bert_tokens)))

        bert_label_start, bert_label_end = [], []
        bert_seq_width = []
        bert_rule = []
        cur_label_start, cur_label_end = [], []
        i = sum(seq_width[:orig_start])
        ptr = orig_start
        rem, sw, cur_len, cur_rule, ptr = self.upd_by_ptr(seq_width, curr_len_list, rule_seq, ptr)
        while i < len(label_start):
            if rem > 0:
                st, ed = label_start[i], label_end[i]
                i += 1
                start = st + cum_num_list[st] if st < len(cum_num_list) else st
                end = ed + cum_num_list[ed] + curr_len_list[ed] - 1 if ed < len(cum_num_list) else ed
                if start >= new_len or end >= new_len:
                    sw = max(1, sw - 1)
                    start, end = 0, 0
                zeros = [0] * (cur_len - 1)
                cur_label_start.append([start] + zeros)
                cur_label_end.append([end] + zeros)
                rem -= 1
            if rem == 0:
                bert_seq_width.extend([sw] * cur_len)
                bert_rule.extend([cur_rule] * cur_len)
                for tup_s, tup_e in zip(zip(*cur_label_start), zip(*cur_label_end)):
                    bert_label_start.append(tup_s)
                    bert_label_end.append(tup_e)
                cur_label_start.clear()
                cur_label_end.clear()
                if ptr < len(curr_len_list):
                    rem, sw, cur_len, cur_rule, ptr = self.upd_by_ptr(seq_width, curr_len_list, rule_seq, ptr)
                else:
                    assert(i == len(label_start))
        assert(len(bert_label_start) == len(bert_seq_width) == len(bert_rule))
        return bert_tokens, bert_label_action[src_start:], bert_label_start, bert_label_end, bert_seq_width, bert_rule, source_indices

    def _split_multi_span(self, seq):
        sid = 0
        seq_out = [sid]
        seq_width = [1]
        for si, i in enumerate(seq):
            if ',' in i:
                slst = list(map(self.to_int, i.split(',')))[:self.max_sp_len]
                seq_out.extend(slst)
                seq_width.append(len(slst))
            else:
                seq_out.append(self.to_int(i))
                seq_width.append(1)
        return seq_out, seq_width

    def get_sens_tags(self, line):
        line1, line2 = line
        src, tgt = line1.split("\t")
        tgt = ' '.join(tgt.strip().split())
        tokens = [self.tokenizer.cls_token] + src.strip().split(' ')

        action_seq, span_seq, rule_seq = zip(*[x.split('|') for x in\
                line2.strip().split(' ')])
        start_seq, end_seq = zip(*[x.split('#') for x in span_seq])
        action_seq = [self.tag2idx.get(tag) for tag in ('DELETE',) + action_seq]
        start_seq, seq_width = self._split_multi_span(start_seq)
        end_seq, _ = self._split_multi_span(end_seq)
        rule_seq = [0] + list(map(self.to_int, rule_seq))

        bert_tokens, bert_label_action, bert_label_start, bert_label_end, bert_seq_width, bert_rule, src_indices = self._split_to_wordpieces_span(tokens, action_seq, start_seq, end_seq, seq_width, rule_seq)
        sentence = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        return sentence, bert_label_action, bert_label_start, bert_label_end, bert_seq_width, bert_rule, tgt, src_indices

    def load_sentences_tags(self, sentences_file, tags_file, d, n_proc=4, rng=None):
        """Loads sentences and tags from their corresponding files.
        Maps tokens and tags to their indices and stores them in the provided dict d.
        """
        with open(sentences_file, 'r') as sen_f, open(tags_file, 'r') as tag_f:
            inp = list(zip(sen_f.readlines(), tag_f.readlines()))
            if rng is not None:
                assert(len(rng) == 2)
                inp[:] = inp[rng[0]:rng[1]]
        with Pool(n_proc) as p:
            out = p.map(self.get_sens_tags, inp)
        d['data'], d['action'], d['start'], d['end'], d['sp_width'], d['rule'], d['ref'], d['src_idx'] = zip(*out)
        d['size'] = len(d['data'])
        assert len(d['data']) == len(d['action'])

    def load_data(self, data_type, rng=None, domain_suf=''):
        """Loads the data for each type in types from data_dir.

        Args:
            data_type: (str) has one of 'train', 'val', 'test' depending on which data is required.
        Returns:
            data: (dict) contains the data with tags for each type in types.
        """
        data = {}
        allowed = ['train', 'dev', 'test']
        if data_type in allowed:
            sentences_file = os.path.join(self.data_dir, data_type, f'sentences{domain_suf}.txt')
            tags_path = os.path.join(self.data_dir, data_type, f'tags{domain_suf}.txt')
            self.load_sentences_tags(sentences_file, tags_path, data, rng=rng)
        else:
            raise ValueError(f"data type not in {allowed}")
        return data

    @staticmethod
    def copy_data(batch_len, max_subwords_len, tags, pad):
        batch_tags = pad * np.ones((batch_len, max_subwords_len))
        for j in range(batch_len):
            tlen = min(len(tags[j]), max_subwords_len)
            batch_tags[j][:tlen] = tags[j][:tlen]
        return batch_tags

    @staticmethod
    def copy_data_3d(batch_len, max_subwords_len, tags, pad, max_sp_len):
        batch_tags = np.full((batch_len, max_subwords_len, max_sp_len), pad)
        for j in range(batch_len):
            tlen = min(len(tags[j]), max_subwords_len)
            for k in range(tlen):
                batch_tags[j][k][:len(tags[j][k])] = tags[j][k]
        return batch_tags

    def to_device(self, data, dtype=torch.long):
        return torch.tensor(data, dtype=dtype).to(self.device)

    def data_iterator(self, data, shuffle=False):
        """Returns a generator that yields batches data with tags.

        Args:
            data: (dict) contains data which has keys 'data', 'tags' and 'size'
            shuffle: (bool) whether the data should be shuffled

        Yields:
            batch_data: (tensor) shape: (batch_size, max_len)
            batch_tags: (tensor) shape: (batch_size, max_len, max_sp_len)
        """

        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        order = list(range(data['size']))
        if shuffle:
            random.seed(self.seed)
            random.shuffle(order)

        if data['size'] % self.batch_size == 0:
            BATCH_NUM = data['size']//self.batch_size
        else:
            BATCH_NUM = data['size']//self.batch_size + 1


        # one pass over data
        bis = list(range(0, data['size'], self.batch_size))
        bis.append(len(order))
        for i in range(BATCH_NUM):
            # fetch sentences and tags
            batch_max_sp_len = 0
            sentences, ref, action, start, end, sp_width, rule, src_idx = [], [], [], [], [], [], [], []
            for idx in order[bis[i]:bis[i + 1]]:
                sentences.append(data['data'][idx])
                ref.append(data['ref'][idx])
                action.append(data['action'][idx])
                start.append(data['start'][idx])
                end.append(data['end'][idx])
                sp_width.append(data['sp_width'][idx])
                rule.append(data['rule'][idx])
                src_idx.append(data['src_idx'][idx])
                batch_max_sp_len = max(max(sp_width[-1]), batch_max_sp_len)

            batch_len = len(sentences)

            batch_max_subwords_len = max([len(s) for s in sentences])
            max_subwords_len = min(batch_max_subwords_len, self.max_len)
            max_src_len = min(max(len(s) for s in src_idx), self.max_len)

            batch_data = self.token_pad_idx * np.ones((batch_len, max_subwords_len))
            # copy the data to the numpy array
            for j in range(batch_len):
                cur_subwords_len = len(sentences[j])
                if cur_subwords_len <= max_subwords_len:
                    batch_data[j][:cur_subwords_len] = sentences[j]
                else:
                    batch_data[j] = sentences[j][:max_subwords_len]

            batch_action = self.to_device(self.copy_data(batch_len, max_src_len, action, self.tag_pad_idx))
            batch_start = self.to_device(self.copy_data_3d(batch_len, max_src_len, start, 0, batch_max_sp_len))
            batch_end = self.to_device(self.copy_data_3d(batch_len, max_src_len, end, 0, batch_max_sp_len))
            batch_sp_width = self.to_device(self.copy_data(batch_len, max_src_len, sp_width, 0), dtype=torch.int)
            batch_rule = self.to_device(self.copy_data(batch_len, max_src_len, rule, self.rule_pad_idx))
            batch_src_idx = self.to_device(self.copy_data(batch_len, max_src_len, src_idx, self.token_pad_idx))

            batch_data = self.to_device(batch_data)
            yield batch_data, ref, batch_action, batch_start, batch_end, batch_sp_width, batch_rule, batch_src_idx
