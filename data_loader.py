"""Data loader"""
import os
import torch
import random
import numpy as np

from pathos.multiprocessing import ProcessingPool as Pool
from transformers import BertTokenizer

class DataLoader(object):
    def __init__(self, data_dir, bert_class, params, token_pad_idx=0, tag_pad_idx=-1):
        self.data_dir = data_dir
        self.batch_size = params.batch_size
        self.max_len = params.max_len
        self.device = params.device
        self.seed = params.seed
        self.token_pad_idx = token_pad_idx
        self.tag_pad_idx = tag_pad_idx

        tags = self.load_tags()
        self.tag2idx = {tag: idx for idx, tag in enumerate(tags)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(tags)}
        params.tag2idx = self.tag2idx
        params.idx2tag = self.idx2tag

        self.tokenizer = BertTokenizer.from_pretrained(bert_class, do_lower_case=False)

        self.max_sp_len = 3
        self.to_int = lambda x: int(x) + 1

    def load_tags(self):
        return ["KEEP", "DELETE"]

    def _split_to_wordpieces_span(self, tokens, label_action, label_start, seq_width, label_end):
        bert_tokens = []
        bert_label_action = []
        bert_label_start = []
        bert_label_end = []
        bert_seq_width = []
        source_indices = []

        cum_num_list = []
        curr_len_list = []
        cum_num = 0
        src_start = len(tokens)
        orig_start = len(tokens)
        for i, token in enumerate(tokens):
            if token == '[SEP]':
                pieces = [token]
            else:
                pieces = self.tokenizer.tokenize(token)
                if token == '|':
                    src_start = len(bert_tokens) + 1
                    orig_start = i + 1

            bert_label_action.extend([label_action[i]] * len(pieces))
            bert_tokens.extend(pieces)

            # bugfix for tokenizer influence on label start and label end
            # do you know johnson jackson | yes I know him - > yes I know johnson
            # Delete|0-0 Delete|0-0 Delete|0-0 Delete|0-0 Delete|0-0 Delete|0-0 Keep|0-0 Keep|0-0 Keep|0-0 Delete|3-4
            # do you know john #son jack #son | yes I know him
            # Delete|0-0 Delete|0-0 Delete|0-0 Delete|0-0 Delete|0-0 Delete|0-0 Delete|0-0 Delete|0-0 Keep|0-0 Keep|0-0 Keep|0-0 Delete|3-6
            # curr_len_list = [1,1,1,2,2,1,1,1,1,1]
            # cum_num_list = [0,0,0,0,1,2,2,2,2,2]

            curr_len_list.append(len(pieces))
            cum_num_list.append(cum_num)
            cum_num += len(pieces)-1

        if len(bert_tokens) > self.max_len:
            new_len = self.max_len - (len(bert_tokens) - src_start)
            source_indices = list(range(new_len, self.max_len))
            bert_tokens = bert_tokens[:new_len] + bert_tokens[src_start:]
        else:
            new_len = src_start
            source_indices = list(range(src_start, len(bert_tokens)))

        cur_label_start, cur_label_end = [], []
        target_width = sum(seq_width[:orig_start - 1])
        ptr = orig_start
        rem = sw = seq_width[ptr]
        for i in range(target_width, len(label_start)):
            if rem == 0:
                bert_seq_width.extend([sw] * curr_len_list[ptr])
                for tup in list(zip(*cur_label_start)):
                    bert_label_start.append(tup)
                for tup in list(zip(*cur_label_end)):
                    bert_label_end.append(tup)
                cur_label_start.clear()
                cur_label_end.clear()
                ptr += 1
                if ptr == len(seq_width):
                    break
                rem = sw = seq_width[ptr]
            st, ed = label_start[i], label_end[i]
            start = st + cum_num_list[st] if st < len(cum_num_list) else st
            end = ed + cum_num_list[ed] + curr_len_list[ed] - 1 if ed < len(cum_num_list) else ed
            if start >= new_len or end >= new_len:
                sw -= 1
                if sw == 0:
                    sw = 1
                    zeros = [0] * curr_len_list[ptr]
                    cur_label_start.append(zeros)
                    cur_label_end.append(zeros)
            else:
                zeros = [0] * (curr_len_list[ptr] - 1)
                cur_label_start.append([start] + zeros)
                cur_label_end.append([end] + zeros)
            rem -= 1
        return bert_tokens, bert_label_action[src_start:], bert_label_start, bert_label_end, bert_seq_width, source_indices

    def _split_multi_span(self, seq):
        sid = 0
        seq_out = [sid]
        seq_width = [1]
        for si, i in enumerate(seq):
            if ',' in i:
                slst = list(map(self.to_int, i.split(',')))
                if len(slst) < self.max_sp_len - 1:
                    slst = slst[:self.max_sp_len-1] + [sid]
                else:
                    slst = slst[:self.max_sp_len]
                seq_out.extend(slst)
                seq_width.append(len(slst))
            else:
                seq_out.append(self.to_int(i))
                sw = 1
                if seq_out[-1] > 0:
                    seq_out.append(sid)
                    sw += 1
                seq_width.append(sw)
        return seq_out, seq_width

    def get_sens_tags(self, line):
        line1, line2 = line
        src, tgt = line1.split("\t")
        tgt = " ".join(tgt.strip().split())
        tokens = src.strip().split(' ')
        seq = line2.strip().split(' ')
        action_seq = [k.split("|")[0] for k in seq]
        start_seq = [k.split("|")[1].split("#")[0] for k in seq]
        end_seq = [k.split("|")[1].split("#")[1] for k in seq]
        action_seq = ['DELETE'] + action_seq
        action_seq = [self.tag2idx.get(tag) for tag in action_seq]
        tokens = ['[CLS]'] + tokens
        start_seq, seq_width = self._split_multi_span(start_seq)
        end_seq, _ = self._split_multi_span(end_seq)
        bert_tokens, bert_label_action, bert_label_start, bert_label_end, bert_seq_width, src_indices = self._split_to_wordpieces_span(tokens, action_seq, start_seq, seq_width, end_seq)
        sentence = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        return sentence, bert_label_action, bert_label_start, bert_seq_width, bert_label_end, tgt, src_indices

    def load_sentences_tags(self, sentences_file, tags_file, d, n_proc=4):
        """Loads sentences and tags from their corresponding files.
        Maps tokens and tags to their indices and stores them in the provided dict d.
        """
        with open(sentences_file, 'r') as file1:
            with open(tags_file, 'r') as file2:
                inp = list(zip(file1.readlines(), file2.readlines()))
        with Pool(n_proc) as p:
            out = p.map(self.get_sens_tags, inp)
        #out = [self.get_sens_tags(x) for x in inp[:100]]
        d['data'], d['action'], d['start'], d['sp_width'], d['end'], d['ref'], d['src_idx'] = zip(*out)
        d['size'] = len(d['data'])
        assert len(d['data']) == len(d['action'])

    def load_data(self, data_type):
        """Loads the data for each type in types from data_dir.

        Args:
            data_type: (str) has one of 'train', 'val', 'test' depending on which data is required.
        Returns:
            data: (dict) contains the data with tags for each type in types.
        """
        data = {}
        allowed = ['train', 'dev', 'test']
        if data_type in allowed:
            sentences_file = os.path.join(self.data_dir, data_type, 'sentences.txt')
            tags_path = os.path.join(self.data_dir, data_type, 'tags.txt')
            self.load_sentences_tags(sentences_file, tags_path, data)
        elif data_type == 'interactive':
            sentences_file = os.path.join(self.data_dir, data_type, 'sentences.txt')
            self.load_sentences_tags(sentences_file, tags_file=None, d=data)
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
            sentences, ref, action, start, end, sp_width, src_idx = [], [], [], [], [], [], []
            for idx in order[bis[i]:bis[i + 1]]:
                sentences.append(data['data'][idx])
                ref.append(data['ref'][idx])
                action.append(data['action'][idx])
                start.append(data['start'][idx])
                end.append(data['end'][idx])
                sp_width.append(data['sp_width'][idx])
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
            batch_src_idx = self.to_device(self.copy_data(batch_len, max_src_len, src_idx, self.token_pad_idx))

            batch_data = self.to_device(batch_data)
            yield batch_data, ref, batch_action, batch_start, batch_end, batch_sp_width, batch_src_idx
