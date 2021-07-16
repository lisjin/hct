import codecs
import json
import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import *
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from torch.distributions import Categorical
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
cc = SmoothingFunction()

from utils import tags_to_string
from multi_headed_additive_attn import MultiHeadedAttention
from data_preprocess_en import utils as dutils

# span classifier based on self-attention
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
        attn_w0 = amask / torch.clamp(amask.sum(2, keepdim=True), min=1e-6)
        sts, eds, masks = [attn_w0], [attn_w0], []
        hid1, hid2 = src_hid, src_hid
        for i in range(max_sp_len):
            mask = (i < sp_width).float()
            masks.append(mask)
            mask = mask.unsqueeze(-1) * amask
            hid1 = self.upd_hid(sts[-1], hid, hid1)
            sts.append(self.span_st_attn(hid, hid, hid1, mask=mask)) # [batch, seq, seq]
            hid2 = self.upd_hid(eds[-1], hid, hid2)
            eds.append(self.span_ed_attn(hid, hid, hid2, mask=mask)) # [batch, seq, seq]
        return torch.stack(sts[1:], -2), torch.stack(eds[1:], -2), torch.stack(masks, -1)

# dist: [batch, seq, seq]
# refs: [batch, seq, seq]
# masks: [batch, seq]
def token_classification_loss_v2(dist, refs, masks):
    loss = torch.sum(dist.log() * refs.float(), dim=-1) # [batch, seq]
    num_tokens = torch.sum(masks).item()
    #assert num_tokens > 1
    return -1.0 * torch.sum(loss * masks) / num_tokens if num_tokens > 0 else torch.sum(loss * 0.0)

# start_dist: [batch, seq, seq]
# end_dist: [batch, seq, seq]
# start_positions: [batch, seq, seq]
# end_positions: [batch, seq, seq]
# seq_masks: [batch, seq]
def span_loss(start_dist, end_dist, start_positions, end_positions, seq_masks):
    span_st_loss = token_classification_loss_v2(start_dist, start_positions, seq_masks)
    span_ed_loss = token_classification_loss_v2(end_dist, end_positions, seq_masks)
    return span_st_loss + span_ed_loss


def clip_and_normalize(word_probs, epsilon):
    word_probs = torch.clamp(word_probs, epsilon, 1.0 - epsilon)
    return word_probs / word_probs.sum(dim=-1, keepdim=True)


def convert_tokens_to_string(tokens):
    """ Converts a sequence of tokens (string) in a single string. """
    out_string = " ".join(tokens).replace(" ##", "").strip()
    return out_string


def decode_into_string(source, label_action, label_start, label_end, label_mask,
        context=None):
    assert len(source) == len(label_action)

    labels = []
    action_map = {0:"KEEP", 1:"DELETE"}
    for idx in range(0, len(label_action)):
        if (idx < len(label_action) and label_mask[idx]):
            if label_end[idx] == 0 or label_start[idx] > label_end[idx]:
                st = 0
                ed = 0
            else:
                st = dutils.ilst2str(label_start[idx])
                ed = dutils.ilst2str(label_end[idx])
            labels.append(action_map[label_action[idx]]+"|"+str(st)+"#"+str(ed))
        else:
            labels.append('DELETE')
    output_tokens = tags_to_string(source, labels, context=context)
    return convert_tokens_to_string(output_tokens)


class BertForSequenceTagging(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceTagging, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.span_classifier = SpanClassifier(config.hidden_size, 0.0)
        self._rl_ratio = 0.5
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #self._tokenizer = BertTokenizer.from_pretrained("./dialogue_model/")

        self.init_weights()

    def forward(self, input_data, rl_model, attention_mask, labels_action,
            labels_start, labels_end, sp_width, src_idx):
        input_ids, input_ref = input_data
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        seq_output = outputs[0]
        src_mask = src_idx.gt(0)
        src_output = seq_output.gather(1, src_idx.unsqueeze(2).expand(-1, -1,
            seq_output.shape[2])) * src_mask.unsqueeze(2).to(seq_output.dtype)
        src_idx = input_ids.gather(1, src_idx) * src_mask.long()

        if rl_model == None:
            self._rl_ratio = 0.0
        logits = self.classifier(src_output) #[bs, seq, 2]

        max_sp_len = labels_start.shape[-1]
        start_dist, end_dist, loss_mask = self.span_classifier(seq_output, sp_width, max_sp_len, attention_mask, src_output, src_mask)
        start_outputs = start_dist.argmax(dim=-1) # [batch, seq]
        end_outputs = end_dist.argmax(dim=-1) # [batch, seq]

        outputs = (logits, start_outputs, end_outputs)
        num_classes = seq_output.shape[1]
        labels_start = F.one_hot(labels_start, num_classes) #[bs, seq, sp_len, seq]
        labels_end = F.one_hot(labels_end, num_classes) #[bs, seq, sp_len, seq]
        loss_span = span_loss(start_dist, end_dist, labels_start, labels_end, loss_mask.float())

        loss_fct = CrossEntropyLoss()
        # Only keep active parts of the loss
        loss_mask = labels_action.gt(-1)
        active_loss = loss_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)[active_loss]
        active_labels = labels_action.view(-1)[active_loss]
        loss = loss_fct(active_logits, active_labels) + loss_span

        if self._rl_ratio > 0.0:
            bsz, seq_len, _, full_len = start_dist.shape
            perm = (0, 2, 1, 3)
            nview = (bsz, max_sp_len, seq_len, 1)
            start_dist = start_dist.permute(*perm).reshape(-1, seq_len, full_len)
            end_dist = end_dist.permute(*perm).reshape(-1, seq_len, full_len)
            samples_action = Categorical(logits=logits).sample() # [bs, seq]
            samples_start = Categorical(logits=start_dist).sample() # [bs, seq]
            samples_end = Categorical(logits=end_dist).sample() # [bs, seq]
            samples_action_prob = torch.gather(logits, 2, samples_action.unsqueeze(dim=2)) #[bs, seq_len, 1]
            samples_start_prob = torch.gather(start_dist, 2, samples_start.unsqueeze(dim=2)) #[bs, seq_len, 1]
            samples_end_prob = torch.gather(end_dist, 2, samples_end.unsqueeze(dim=2)) #[bs, seq_len, 1]

            samples_start = samples_start.view(*nview[:-1]).permute(*perm[:-1])
            samples_end = samples_end.view(*nview[:-1]).permute(*perm[:-1])
            samples_start_prob = samples_start_prob.view(*nview).permute(*perm)
            samples_end_prob = samples_end_prob.view(*nview).permute(*perm)

            samples_action_prob = samples_action_prob.unsqueeze(dim=2)
            samples_start_prob = samples_start_prob.unsqueeze(dim=-1)
            samples_end_prob = samples_end_prob.unsqueeze(dim=-1)

            greedy_action = logits.argmax(dim=-1) # [bs, seq]
            greedy_start = start_outputs # [bs, seq]
            greedy_end = end_outputs # [bs, seq]

            rewards = []
            samples_mask = loss_mask.gt(-1).float()

            def gpt_score(sentence):
                tokenize_input = self._tokenizer.tokenize(sentence)
                if len(tokenize_input)>300:
                    tokenize_input = tokenize_input[:300]
                tensor_input = torch.tensor([self._tokenizer.convert_tokens_to_ids(tokenize_input)])
                tensor_input = tensor_input.cuda()
                outputs = gpt_model(input_ids=tensor_input, labels=tensor_input)
                loss = outputs[0]
                if math.exp(loss) >0.0:
                    ppl = loss
                else:
                    return 0.0
                b = 5.92+3*1.84
                a = 5.92-3*1.84
                #b = 6.24+3*1.99
                #a = 6.24-3*1.99
                if ppl > b:
                    ppl_norm = 1.0
                elif ppl < a:
                    ppl_norm = 0.0
                else:
                    ppl_norm = (b-ppl)/(b-a)
                return ppl_norm

            for i in range(len(samples_start)):
                weight = (0.25, 0.25, 0.25, 0.25)
                input_tokens = self.tokenizer.convert_ids_to_tokens(src_idx[i].tolist())
                sample_str = decode_into_string(input_tokens, samples_action[i].tolist(), samples_start[i].tolist(), samples_end[i].tolist(), samples_mask[i].tolist())
                greedy_str = decode_into_string(input_tokens, greedy_action[i].tolist(), greedy_start[i].tolist(), greedy_end[i].tolist(), samples_mask[i].tolist())
                if type(rl_model) is str:
                    sample_score = sentence_bleu([input_ref[i].split()], sample_str.split(), weights=weight, smoothing_function=cc.method3)
                    greedy_score = sentence_bleu([input_ref[i].split()], greedy_str.split(), weights=weight, smoothing_function=cc.method3)
                elif rl_model is not None:
                    sample_score = gpt_score(sample_str)
                    greedy_score = gpt_score(greedy_str)
                rewards.append(sample_score-greedy_score)

            rewards = torch.tensor(rewards).cuda()

            loss_action_rl = -1.0 * clip_and_normalize(samples_action_prob, 1e-6).log()*rewards.unsqueeze(dim=1)*samples_mask
            loss_action_rl = loss_action_rl.sum()/samples_mask.sum()
            loss_st_rl = -1.0 * clip_and_normalize(samples_start_prob, 1e-6).log()*rewards.unsqueeze(dim=1)*samples_mask
            loss_st_rl = loss_st_rl.sum()/samples_mask.sum()
            loss_ed_rl = -1.0 * clip_and_normalize(samples_end_prob, 1e-6).log()*rewards.unsqueeze(dim=1)*samples_mask
            loss_ed_rl = loss_ed_rl.sum()/samples_mask.sum()

            loss_rl = loss_action_rl + loss_st_rl + loss_ed_rl
            loss = (1.0 - self._rl_ratio) * loss + self._rl_ratio * loss_rl

        outputs = (loss,) + outputs
        return outputs  # (loss), scores
