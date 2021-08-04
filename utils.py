import os
import json
import torch
import torch.nn.functional as F
import logging
import numpy as np

from torch.distributions import Categorical
from transformers import BertConfig

from data_preprocess_en import utils as dutils


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def load_checkpoint(model, ckpt_dir, device, optimizer=None, scheduler=None):
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'model.bin'), map_location=device))
    osd_path = os.path.join(ckpt_dir, 'optim.bin')
    if optimizer is not None and os.path.isfile(osd_path):
        optimizer.load_state_dict(torch.load(osd_path))
    ssd_path = os.path.join(ckpt_dir, 'sched.bin')
    if scheduler is not None and os.path.isfile(ssd_path):
        scheduler.load_state_dict(torch.load(ssd_path))
    best_val_bleu = 0.
    for x in os.listdir(ckpt_dir):
        if x.startswith('pred_dev'):
            best_val_bleu = float(x.split('_')[-1].split('.txt')[0])
    return model, optimizer, scheduler, best_val_bleu


def save_checkpoint(model, ckpt_dir, optimizer, scheduler):
    os.mkdir(ckpt_dir)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'model.bin'))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, 'optim.bin'))
    torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, 'sched.bin'))


def convert_tokens_to_string(tokens):
    """ Converts a sequence of tokens (string) in a single string. """
    return ' '.join(tokens).replace(' ##', '').strip()


def filter_spans(starts, ends, max_i, stop_i=0):
    for i, start in enumerate(starts):
        end = ends[i]
        if start == stop_i or start > end or start >= max_i:
            starts[i] = ends[i] = -1
            continue
    starts = [s for s in starts if s > -1]
    ends = [e for e in ends if e > -1]
    assert(len(starts) == len(ends))
    return starts, ends


def get_sp_strs(start_lst, end_lst, context_len):
    max_i = context_len - 1
    starts, ends = filter_spans(start_lst, end_lst, max_i)
    if not starts:
        starts.append(0)
        ends.append(0)
    starts, ends = dutils.ilst2str(starts), dutils.ilst2str(ends)
    return starts, ends


def load_rules(rule_path, mask='_', fmask='{}'):
    with open(rule_path, encoding='utf8') as f:
        rules = [''] + [l.strip().replace(mask, fmask) for l in f]
    rule_slot_cnts = [sum(int(y == fmask) for y in x.split()) for x in rules]
    return rules, rule_slot_cnts


def get_config(params, bert_class, bleu_rl):
    config = BertConfig.from_pretrained(bert_class)
    config.bert_class = bert_class
    config.device = params.device
    config.rl_model = 'bleu' if bleu_rl else None
    config.rl_ratio = params.rl_ratio

    config.num_labels = len(params.tag2idx)
    config.tags = params.idx2tag
    config.pad_tag_id = params.pad_tag_id

    config.rules = params.rules
    config.rule_slot_cnts = params.rule_slot_cnts
    config.max_sp_len = params.max_sp_len
    config.additional_special_tokens = tuple(f'[SLOT{x}]' for x in range(params.max_sp_len))
    config.vocab_size += len(config.additional_special_tokens)
    return config


def safe_log(inp, eps=1e-45):
    return (inp + eps).log()


def sample_helper(logits, dim=2):
    samples = Categorical(logits=logits).sample()
    samples_prob = torch.gather(logits, dim, samples.unsqueeze(dim))
    return samples, samples_prob


def cls_loss(dist, refs, masks):
    refs = F.one_hot(refs, dist.shape[-1])
    loss = torch.sum(safe_log(dist) * refs.float(), dim=-1)
    num_tokens = torch.sum(masks).item()
    return -torch.sum(loss * masks) / num_tokens
