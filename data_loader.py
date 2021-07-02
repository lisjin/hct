"""Data loader"""
import os
import torch
import utils
import random
import numpy as np
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


    def load_tags(self):
        return ["KEEP", "DELETE"]

    def _split_to_wordpieces_span(self, tokens, label_action, label_start, label_end):

        bert_tokens = []
        bert_label_action = []
        bert_label_start = []
        bert_label_end = []
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

        for i in range(len(label_start)):
            st = label_start[i]
            ed = label_end[i]
            bert_label_start.extend([st+cum_num_list[st]] + [0]*(curr_len_list[i]-1))
            bert_label_end.extend([ed+cum_num_list[ed]+curr_len_list[ed]-1] + [0]*(curr_len_list[i]-1))

        return bert_tokens, bert_label_action, bert_label_start, bert_label_end, token_start_indices

    def load_sentences_tags(self, sentences_file, tags_file, d):
        """Loads sentences and tags from their corresponding files.
            Maps tokens and tags to their indices and stores them in the provided dict d.
        """
        sentences = []
        ref = []
        action = []
        start = []
        end = []
        with open(sentences_file, 'r') as file1:
            with open(tags_file, 'r') as file2:
                for line1, line2 in list(zip(file1,file2)):
                    src, tgt = line1.split("\t")
                    tgt = " ".join(tgt.strip().split())
                    ref.append(tgt.lower())
                    tokens = src.strip().split(' ')
                    seq = line2.strip().split(' ')
                    action_seq = [k.split("|")[0] for k in seq]
                    start_seq = [k.split("|")[1].split("#")[0] for k in seq]
                    end_seq = [k.split("|")[1].split("#")[1] for k in seq]
                    action_seq = [self.tag2idx.get(tag) for tag in action_seq]

                    tokens = ['[CLS]'] + tokens
                    action_seq = [1] + action_seq
                    start_seq = [0] + [int(i)+1 for i in start_seq]
                    end_seq = [0] + [int(i)+1 for i in end_seq]
                    bert_tokens, bert_label_action, bert_label_start, bert_label_end, token_start_idxs = self._split_to_wordpieces_span(tokens, action_seq, start_seq, end_seq)
                    sentences.append((self.tokenizer.convert_tokens_to_ids(bert_tokens), token_start_idxs))
                    action.append(bert_label_action)
                    start.append(bert_label_start)
                    end.append(bert_label_end)
            # checks to ensure there is a tag for each token
            assert len(sentences) == len(action)
            #for i in range(len(sentences)):
            #    assert len(action[i]) == len(sentences[i][-1])

            d['action'] = action
            d['start'] = start
            d['end'] = end
            d["ref"] = ref

        # storing sentences and tags in dict d
        d['data'] = sentences
        d['size'] = len(sentences)

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
            print('Loading ' + data_type)
            sentences_file = os.path.join(self.data_dir, data_type, 'sentences.txt')
            tags_path = os.path.join(self.data_dir, data_type, 'tags.txt')
            self.load_sentences_tags(sentences_file, tags_path, data)
        elif data_type == 'interactive':
            sentences_file = os.path.join(self.data_dir, data_type, 'sentences.txt')
            self.load_sentences_tags(sentences_file, tags_file=None, d=data)
        else:
            raise ValueError(f"data type not in {allowed}")
        return data

    def data_iterator(self, data, shuffle=False):
        """Returns a generator that yields batches data with tags.

        Args:
            data: (dict) contains data which has keys 'data', 'tags' and 'size'
            shuffle: (bool) whether the data should be shuffled

        Yields:
            batch_data: (tensor) shape: (batch_size, max_len)
            batch_tags: (tensor) shape: (batch_size, max_len)
        """

        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        order = list(range(data['size']))
        if shuffle:
            random.seed(self.seed)
            random.shuffle(order)

        interMode = False if 'action' in data else True

        if data['size'] % self.batch_size == 0:
            BATCH_NUM = data['size']//self.batch_size
        else:
            BATCH_NUM = data['size']//self.batch_size + 1


        # one pass over data
        for i in range(BATCH_NUM):
            # fetch sentences and tags
            if i * self.batch_size < data['size'] < (i+1) * self.batch_size:
                sentences = [data['data'][idx] for idx in order[i*self.batch_size:]]
                ref = [data['ref'][idx] for idx in order[i*self.batch_size:]]
                if not interMode:
                    action = [data['action'][idx] for idx in order[i*self.batch_size:]]
                    start = [data['start'][idx] for idx in order[i*self.batch_size:]]
                    end = [data['end'][idx] for idx in order[i*self.batch_size:]]
            else:
                sentences = [data['data'][idx] for idx in order[i*self.batch_size:(i+1)*self.batch_size]]
                ref = [data['ref'][idx] for idx in order[i*self.batch_size:(i+1)*self.batch_size]]
                if not interMode:
                    action = [data['action'][idx] for idx in order[i*self.batch_size:(i+1)*self.batch_size]]
                    start = [data['start'][idx] for idx in order[i*self.batch_size:(i+1)*self.batch_size]]
                    end = [data['end'][idx] for idx in order[i*self.batch_size:(i+1)*self.batch_size]]

            # batch length
            batch_len = len(sentences)

            # compute length of longest sentence in batch
            batch_max_subwords_len = max([len(s[0]) for s in sentences])
            max_subwords_len = min(batch_max_subwords_len, self.max_len)
            max_token_len = 0


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
                max_token_len = max(int(sum(token_starts)), max_token_len)

            if not interMode:
                def copy_data(batch_len, max_token_len, tags, pad):
                    batch_tags = pad * np.ones((batch_len, max_token_len))
                    for j in range(batch_len):
                        cur_tags_len = len(tags[j])
                        if cur_tags_len <= max_token_len:
                            batch_tags[j][:cur_tags_len] = tags[j]
                        else:
                            batch_tags[j] = tags[j][:max_token_len]
                    return batch_tags

                batch_action = copy_data(batch_len, max_subwords_len, action, self.tag_pad_idx)
                batch_start = copy_data(batch_len, max_subwords_len, start, 0)
                batch_end = copy_data(batch_len, max_subwords_len, end, 0)

            # since all data are indices, we convert them to torch LongTensors
            batch_data = torch.tensor(batch_data, dtype=torch.long)
            batch_token_starts = torch.tensor(batch_token_starts, dtype=torch.long)
            if not interMode:
                batch_action = torch.tensor(batch_action, dtype=torch.long)
                batch_start = torch.tensor(batch_start, dtype=torch.long)
                batch_end = torch.tensor(batch_end, dtype=torch.long)

            # shift tensors to GPU if available
            batch_data, batch_token_starts = batch_data.to(self.device), batch_token_starts.to(self.device)
            if not interMode:
                batch_action = batch_action.to(self.device)
                batch_start = batch_start.to(self.device)
                batch_end = batch_end.to(self.device)
                yield batch_data, batch_token_starts, ref, batch_action, batch_start, batch_end
            else:
                yield batch_data, batch_token_starts
