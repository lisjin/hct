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

    def _split_to_wordpieces_span(self, tokens, label_action, label_start, label_lens, label_end, seq_width):

        bert_tokens = []
        bert_label_action = []
        bert_label_start = []
        bert_label_end = []
        bert_seq_width = []
        token_start_indices = []

        cum_num_list = []
        curr_len_list = []
        cum_num = 0
        for i, token in enumerate(tokens):
            token_start_indices.append(len(bert_tokens) + 1)
            if token == "[SEP]":
                pieces = ['[SEP]']
            else:
                pieces = self.tokenizer.tokenize(token)

            bert_tokens.extend(pieces)
            bert_label_action.extend([label_action[i]] * len(pieces))
            bert_seq_width.extend([seq_width[i]] * len(pieces))

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

        i2, rem = -1, 0
        for i in range(len(label_start)):
            if rem == 0:
                i2 += 1
                rem = label_lens[i2]
            st = label_start[i]
            ed = label_end[i]
            zeros = [0] * (curr_len_list[i2] - 1)
            bert_label_start.extend([(st+cum_num_list[st] if st < len(cum_num_list) else st)] + zeros)
            bert_label_end.extend([(ed+cum_num_list[ed]+curr_len_list[ed]-1 if ed < len(cum_num_list) else ed)] + zeros)
            rem -= 1

        return bert_tokens, bert_label_action, bert_label_start, bert_label_end, bert_seq_width, token_start_indices

    def _split_multi_span(self, seq):
        sid = 0
        seq_out = [sid]
        seq_width = [1]
        for si, i in enumerate(seq):
            if ',' in i:
                slst = list(map(self.to_int, i.split(',')))
                slst = slst[:self.max_sp_len] + [sid] * int(len(slst) < self.max_sp_len)
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
        action_seq = [self.tag2idx.get(tag) for tag in action_seq]
        tokens = ['[CLS]'] + tokens
        action_seq = [self.tag2idx.get('DELETE')] + action_seq
        start_seq, seq_width = self._split_multi_span(start_seq)
        end_seq, _ = self._split_multi_span(end_seq)
        bert_tokens, bert_label_action, bert_label_start, bert_label_end, bert_seq_width, token_start_idxs = self._split_to_wordpieces_span(tokens, action_seq, start_seq, seq_width, end_seq, seq_width)
        sentence = (self.tokenizer.convert_tokens_to_ids(bert_tokens), token_start_idxs)
        return sentence, bert_label_action, bert_label_start, bert_seq_width, bert_label_end, tgt

    def load_sentences_tags(self, sentences_file, tags_file, d, n_proc=4):
        """Loads sentences and tags from their corresponding files.
            Maps tokens and tags to their indices and stores them in the provided dict d.
        """
        with open(sentences_file, 'r') as file1:
            with open(tags_file, 'r') as file2:
                inp = list(zip(file1.readlines(), file2.readlines()))
        with Pool(n_proc) as p:
            out = p.map(self.get_sens_tags, inp)
        d['data'], d['action'], d['start'], d['sp_width'], d['end'], d['ref'] = zip(*out)
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
    def copy_data_3d(batch_len, max_subwords_len, tags, pad, sp_width, max_sp_len):
        batch_tags = np.full((batch_len, max_subwords_len, max_sp_len), pad)
        for j in range(batch_len):
            k2, rem = -1, 0
            tlen = min(len(tags[j]), max_subwords_len)
            for k in range(tlen):
                if rem == 0:
                    k2 += 1
                    rem = sp_width[j][k2]
                    batch_tags[j][k2][:rem] = tags[j][k:k+rem]
                rem -= 1
        return batch_tags

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
            sentences, ref, action, start, end, sp_width = [], [], [], [], [], []
            for idx in order[bis[i]:bis[i + 1]]:
                sentences.append(data['data'][idx])
                ref.append(data['ref'][idx])
                action.append(data['action'][idx])
                start.append(data['start'][idx])
                end.append(data['end'][idx])
                sp_width.append(data['sp_width'][idx])
                batch_max_sp_len = max(max(sp_width[-1]), batch_max_sp_len)

            # batch length
            batch_len = len(sentences)

            # compute length of longest sentence in batch
            batch_max_subwords_len = max([len(s[0]) for s in sentences])
            max_subwords_len = min(batch_max_subwords_len, self.max_len)

            # prepare a numpy array with the data, initialising the data with pad_idx
            batch_data = self.token_pad_idx * np.ones((batch_len, max_subwords_len))
            batch_token_starts = []

            # copy the data to the numpy array
            for j in range(batch_len):
                cur_subwords_len = len(sentences[j][0])
                if cur_subwords_len <= max_subwords_len:
                    batch_data[j][:cur_subwords_len] = sentences[j][0]
                else:
                    batch_data[j] = sentences[j][0][:max_subwords_len]
                token_start_idx = sentences[j][-1]
                token_starts = np.zeros(max_subwords_len)
                token_starts[[idx for idx in token_start_idx if idx < max_subwords_len]] = 1
                batch_token_starts.append(token_starts)

            batch_action = self.copy_data(batch_len, max_subwords_len, action, self.tag_pad_idx)
            batch_start = self.copy_data_3d(batch_len, max_subwords_len, start, 0, sp_width, batch_max_sp_len)
            batch_end = self.copy_data_3d(batch_len, max_subwords_len, end, 0, sp_width, batch_max_sp_len)
            batch_sp_width = self.copy_data(batch_len, max_subwords_len, sp_width, 0)

            # since all data are indices, we convert them to torch LongTensors
            batch_data = torch.tensor(batch_data, dtype=torch.long).to(self.device)
            batch_token_starts = torch.tensor(batch_token_starts, dtype=torch.long).to(self.device)
            batch_action = torch.tensor(batch_action, dtype=torch.long).to(self.device)
            batch_start = torch.tensor(batch_start, dtype=torch.long).to(self.device)
            batch_end = torch.tensor(batch_end, dtype=torch.long).to(self.device)
            batch_sp_width = torch.tensor(batch_sp_width, dtype=torch.long).to(self.device)
            yield batch_data, batch_token_starts, ref, batch_action, batch_start, batch_end, batch_sp_width
