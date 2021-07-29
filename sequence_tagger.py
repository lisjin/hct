import codecs
import json
import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from transformers.models.bert.modeling_bert import BertPreTrainedModel, CrossEntropyLoss
from transformers import BertTokenizer, BertModel
from torch.distributions import Categorical
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from utils import convert_tokens_to_string, get_sp_strs

cc = SmoothingFunction()

from utils import tags_to_string
from multi_headed_additive_attn import MultiHeadedAttention


class SpanClassifier(nn.Module):
    def __init__(self, hidden_dim, max_relative_position, dropout=0.1):
        super(SpanClassifier, self).__init__()
        self.span_st_attn = MultiHeadedAttention(1, hidden_dim, max_relative_positions=max_relative_position)
        self.span_ed_attn = MultiHeadedAttention(1, hidden_dim, max_relative_positions=max_relative_position)
        self.sp_emb = nn.Linear(2 * hidden_dim, hidden_dim)
        self.dropout = dropout
        if max_relative_position > 0.0:
            print("Setting max_relative_position to {}".format(max_relative_position))

    def reset_parameters(self):
        nn.init.normal_(self.sp_emb.weight, std=2e-2)
        nn.init.constant_(self.sp_emb.bias, 0.)

    def upd_hid(self, attn_w, hid, hidp):
        hidc = (hid.unsqueeze(1) * attn_w.unsqueeze(-1)).sum(2)
        hidc = F.relu(self.sp_emb(torch.cat((hidp, hidc), 2)), inplace=True)
        hidc = F.dropout(hidc, p=self.dropout, training=self.training)
        return hidc

    def forward(self, hid, sp_width, max_sp_len, attention_mask, src_hid, src_mask):
        amask = torch.logical_and(src_mask.unsqueeze(2), attention_mask.unsqueeze(1)).float()
        attn_d = amask.sum(-1, keepdim=True)
        attn_d.masked_fill_(attn_d == 0, 1.)
        attn_w0 = amask / attn_d
        sts, eds, masks = [attn_w0], [attn_w0], []
        hid1, hid2 = src_hid, src_hid
        for i in range(max_sp_len):
            mask = (i < sp_width).float()
            masks.append(mask)
            mask = torch.logical_and(mask.unsqueeze(-1), amask)
            hid1 = self.upd_hid(sts[-1], hid, hid1)
            hid2 = self.upd_hid(eds[-1], hid, hid2)
            with torch.cuda.amp.autocast(enabled=False):
                hid_f32, mask_f32 = hid.float(), mask.float()
                sts.append(self.span_st_attn(hid_f32, hid_f32, hid1.float(), mask=mask_f32)) # [batch, seq, seq]
                eds.append(self.span_ed_attn(hid_f32, hid_f32, hid2.float(), mask=mask_f32)) # [batch, seq, seq]
        return torch.stack(sts[1:], -2), torch.stack(eds[1:], -2), torch.stack(masks, -1)


def classif_loss(dist, refs, masks):
    refs = F.one_hot(refs, dist.shape[-1])
    loss = torch.sum((dist + 1e-12).log() * refs.float(), dim=-1)
    num_tokens = torch.sum(masks).item()
    return -torch.sum(loss * masks) / num_tokens


def span_loss(start_dist, end_dist, start_positions, end_positions, seq_masks):
    span_st_loss = classif_loss(start_dist, start_positions, seq_masks)
    span_ed_loss = classif_loss(end_dist, end_positions, seq_masks)
    return span_st_loss + span_ed_loss


def clip_and_normalize(word_probs, epsilon):
    word_probs = torch.clamp(word_probs, epsilon, 1.0 - epsilon)
    return word_probs / word_probs.sum(dim=-1, keepdim=True)


def decode_into_string(source, label_action, label_start, label_end, label_mask,
        context=None):
    assert len(source) == len(label_action)
    labels = []
    action_map = {0:"KEEP", 1:"DELETE"}
    stop_i = 0
    if context is None:
        context = source
    context_len = len(context)
    for idx in range(0, len(label_action)):
        st, ed = stop_i, stop_i
        if label_mask[idx]:
            st, ed = get_sp_strs(label_start[idx], label_end[idx], context_len)
            action = action_map[label_action[idx]]
        else:
            action = action_map[1]
        labels.append(f'{action}|{st}#{ed}')
    output_tokens = tags_to_string(source, labels, context=context)
    return convert_tokens_to_string(output_tokens)


class BertForSequenceTagging(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceTagging, self).__init__(config)
        self.num_labels = config.num_labels
        self.rules = config.rules
        self.num_rules = len(self.rules)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.rule_classifier = nn.Linear(config.hidden_size, self.num_rules)
        self.span_classifier = SpanClassifier(config.hidden_size, 0.)

        self.rl_model = config.rl_model
        self.rl_ratio = config.rl_ratio
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_class)
        self.pad_token_id = config.pad_token_id
        self.loss_fn = CrossEntropyLoss()
        self.bleu_fn = partial(sentence_bleu, weights=(.25,) * 4,
                smoothing_function=cc.method3)

        self.init_weights()

    def forward(self, input_data, labels_action, labels_start, labels_end,
            sp_width, rule, src_idx):
        input_ids, input_ref = input_data
        attention_mask = input_ids.ne(self.pad_token_id)
        seq_output = self.bert(input_ids, attention_mask=attention_mask)[0]
        src_mask = src_idx.ne(self.pad_token_id)
        src_output = seq_output.gather(1, src_idx.unsqueeze(2).expand(-1, -1,
            seq_output.shape[2])) * src_mask.unsqueeze(2).to(seq_output.dtype)
        src_idx = input_ids.gather(1, src_idx) * src_mask.long()

        max_sp_len = labels_start.shape[-1]
        start_dist, end_dist, sp_loss_mask = self.span_classifier(seq_output,
                sp_width, max_sp_len, attention_mask, src_output, src_mask)
        sp_loss_mask = sp_loss_mask.float()
        start_outputs = start_dist.argmax(dim=-1)
        end_outputs = end_dist.argmax(dim=-1)

        loss_span = span_loss(start_dist, end_dist, labels_start, labels_end, sp_loss_mask)

        logits = self.classifier(src_output)
        act_loss_mask = labels_action.gt(-1)
        loss_action = self.get_active_loss(act_loss_mask, logits, labels_action, self.num_labels)
        rule_logits = self.rule_classifier(src_output)
        rule_loss_mask = rule.gt(-1)
        loss_rule = self.get_active_loss(rule_loss_mask, rule_logits, rule, self.num_rules)
        loss = loss_action + loss_rule + loss_span

        if self.rl_model is not None:
            loss_rl = self.apply_rl(logits, rule_logits, start_dist, end_dist, start_outputs, end_outputs, src_idx, input_ids, attention_mask, rule_loss_mask, act_loss_mask, sp_loss_mask, input_ref, max_sp_len)
            loss = (1. - self.rl_ratio) * loss + self.rl_ratio * loss_rl

        outputs = (loss, logits, start_outputs, end_outputs)
        return outputs

    def apply_rl(self, logits, rule_logits, start_dist, end_dist, start_outputs, end_outputs,
            src_idx, ctx_idx, attention_mask, act_loss_mask, rule_loss_mask, sp_loss_mask, input_ref, max_sp_len, perm=(0, 2, 1, 3)):
        samples_action = Categorical(logits=logits).sample() # [bs, seq]
        samples_action_prob = torch.gather(logits, 2, samples_action.unsqueeze(dim=2)) #[bs, seq_len, 1]
        greedy_action = logits.argmax(dim=-1) # [bs, seq]

        bsz, seq_len, _, full_len = start_dist.shape
        start_dist = start_dist.permute(*perm).reshape(-1, seq_len, full_len)
        end_dist = end_dist.permute(*perm).reshape(-1, seq_len, full_len)

        samples_start = Categorical(logits=start_dist).sample() # [bs, seq]
        samples_end = Categorical(logits=end_dist).sample() # [bs, seq]
        samples_start_prob = torch.gather(start_dist, 2, samples_start.unsqueeze(dim=2)) #[bs, seq_len, 1]
        samples_end_prob = torch.gather(end_dist, 2, samples_end.unsqueeze(dim=2)) #[bs, seq_len, 1]

        nview = (bsz, max_sp_len, seq_len)
        samples_start = samples_start.view(*nview).permute(*perm[:-1])
        samples_end = samples_end.view(*nview).permute(*perm[:-1])
        samples_start_prob = samples_start_prob.view(*nview).permute(*perm[:-1])
        samples_end_prob = samples_end_prob.view(*nview).permute(*perm[:-1])

        rewards = []
        act_loss_mask = act_loss_mask.float()
        for i in range(len(samples_start)):
            src_len = act_loss_mask[i].long().sum()
            src_tokens = self.tokenizer.convert_ids_to_tokens(src_idx[i][:src_len].tolist())
            ctx_len = attention_mask[i].long().sum()
            ctx_tokens = self.tokenizer.convert_ids_to_tokens(ctx_idx[i][:ctx_len].tolist())
            loss_mask_lst = act_loss_mask[i].tolist()
            sample_str = decode_into_string(src_tokens, samples_action[i][:src_len].tolist(), samples_start[i][:src_len].tolist(), samples_end[i][:src_len].tolist(), loss_mask_lst, context=ctx_tokens)
            greedy_str = decode_into_string(src_tokens, greedy_action[i][:src_len].tolist(), start_outputs[i][:src_len].tolist(), end_outputs[i][:src_len].tolist(), loss_mask_lst, context=ctx_tokens)

            input_ref_lst = [input_ref[i].split()]
            sample_score = self.bleu_fn(input_ref_lst, sample_str.split())
            greedy_score = self.bleu_fn(input_ref_lst, greedy_str.split())
            rewards.append(sample_score - greedy_score)

        rewards = torch.as_tensor(rewards, device=logits.device).unsqueeze(1)
        loss_action_rl = self.rl_loss(samples_action_prob.squeeze(2), act_loss_mask, rewards)
        rewards = rewards.unsqueeze(2)
        loss_st_rl = self.rl_loss(samples_start_prob, sp_loss_mask, rewards)
        loss_ed_rl = self.rl_loss(samples_end_prob, sp_loss_mask, rewards)

        loss_rl = loss_action_rl + loss_st_rl + loss_ed_rl
        return loss_rl

    def get_active_loss(self, loss_mask, logits, labels, num_labels):
        active_loss = loss_mask.view(-1)
        active_logits = logits.view(-1, num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        return self.loss_fn(active_logits, active_labels)

    @staticmethod
    def rl_loss(prob, mask, rewards):
        loss = -clip_and_normalize(prob, 1e-6).log()
        loss *= rewards * mask
        return loss.sum() / mask.sum()
