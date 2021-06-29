from transformers.models.bert.modeling_bert import *
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys, json, codecs

from multi_headed_additive_attn import MultiHeadedAttention
from transformers import BertTokenizer
from torch.distributions import Categorical

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
cc = SmoothingFunction()

import math

# span classifier based on self-attention
class SpanClassifier(nn.Module):
    def __init__(self, hidden_dim, max_relative_position):
        super(SpanClassifier, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.span_st_attn = MultiHeadedAttention(1, hidden_dim, max_relative_positions=max_relative_position)
        self.span_ed_attn = MultiHeadedAttention(1, hidden_dim, max_relative_positions=max_relative_position)
        self.comb_emb = nn.Linear(hidden_dim * 2, hidden_dim)
        if max_relative_position > 0.0:
            print("Setting max_relative_position to {}".format(max_relative_position))

    def reset_parameters(self):
        nn.init.normal_(self.comb_emb.weight, std=2e-2)
        nn.init.constant_(self.comb_emb.bias, 0.)

    def upd_hid(self, attn_w, hid):
        hidc = (hid.unsqueeze(1) * attn_w.unsqueeze(-1)).sum(2)
        hidc = self.comb_emb(torch.cat((hid, hidc), -1))
        hidc = F.relu(hidc, inplace=True)
        return hidc

    def forward(self, hid, sp_width, max_sp_len):
        sts, eds, masks = [], [], []
        hid1, hid2 = hid, hid
        for i in range(max_sp_len):
            masks.append(torch.logical_and(sp_width, i <= sp_width).float())
            sts.append(self.span_st_attn(hid1, hid1, hid1, mask=masks[-1], type="self")) # [batch, seq, seq]
            hid1 = self.upd_hid(sts[-1], hid1)
            eds.append(self.span_ed_attn(hid2, hid2, hid2, mask=masks[-1], type="self")) # [batch, seq, seq]
            hid2 = self.upd_hid(eds[-1], hid2)
        return torch.stack(sts, -2), torch.stack(eds, -2), torch.stack(masks, -1)

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

def decode_into_string(source, label_action, label_start, label_end, label_mask):
    assert len(source) == len(label_action)

    labels = []
    action_map = {0:"KEEP", 1:"DELETE"}

    for idx in range(0, len(label_action)):
        if (idx < len(label_action) and label_mask[idx]):
            if label_end[idx] ==0 or label_start[idx] > label_end[idx]:
                st = 0
                ed = 0
            else:
                st = label_start[idx]
                ed = label_end[idx]
            labels.append(action_map[label_action[idx]]+"|"+str(st)+"#"+str(ed))
        else:
            labels.append('DELETE')
    output_tokens = []
    bad_toks = set(['[SEP]', '*'])
    for token, tag in zip(source, labels):
        if len(tag.split("|"))>1:
            added_phrase = tag.split("|")[1]
            start, end = added_phrase.split("#")[0], added_phrase.split("#")[1]
            if int(end) != 0 and int(end)>=int(start):
                add_phrase = [s for s in source[int(start):int(end)+1] if s not in bad_toks]
                add_phrase = " ".join(add_phrase)
                output_tokens.append(add_phrase)
            if tag.split("|")[0]=="KEEP" and token not in bad_toks:
                output_tokens.append(token)

    output_tokens = " ".join(output_tokens).split()
    while '' in output_tokens:
        output_tokens.remove('')

    if len(output_tokens)==0:
        output_tokens.append("*")
    elif len(output_tokens) > 1 and output_tokens[-1]=="*":
        output_tokens = output_tokens[:-1]
    #return ' '.join(output_tokens).strip()
    return convert_tokens_to_string(output_tokens)


class BertForSequenceTagging(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceTagging, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.span_classifier = SpanClassifier(config.hidden_size, 0.0)
        self.end_span = nn.Embedding(1, config.hidden_size)
        self._rl_ratio = 0.5
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #self._tokenizer = BertTokenizer.from_pretrained("./dialogue_model/")

        self.init_weights()

    def forward(self, input_data, gpt_model, token_type_ids=None,
            attention_mask=None, labels_action=None, labels_start=None,
            labels_end=None, sp_width=None):
        input_ids, input_token_starts, input_ref = input_data
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        if gpt_model == None:
            self._rl_ratio = 0.0
        logits = self.classifier(sequence_output) #[bs, seq, 2]

        # Add last sequence position for end-of-span prediction
        sequence_output = torch.cat((sequence_output,
            self.end_span.weight[None].expand(sequence_output.shape[0], -1, -1)), 1)

        max_sp_len = labels_start.shape[-1]
        start_dist, end_dist, loss_mask = self.span_classifier(sequence_output, sp_width, max_sp_len)
        start_outputs = start_dist.argmax(dim=-1) # [batch, seq]
        end_outputs = end_dist.argmax(dim=-1) # [batch, seq]

        outputs = (logits, start_outputs, end_outputs)
        labels_start = F.one_hot(labels_start, num_classes=labels_start.shape[1]) #[bs, seq + 1, sp_len, seq + 1]
        labels_end = F.one_hot(labels_end, num_classes=labels_start.shape[1]) #[bs, seq + 1, sp_len, seq + 1]
        loss_span = span_loss(start_dist, end_dist, labels_start, labels_end, loss_mask.float())

        loss_fct = CrossEntropyLoss()
        # Only keep active parts of the loss
        active_loss = labels_action.gt(-1).view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)[active_loss]
        active_labels = labels_action.view(-1)[active_loss]
        loss = loss_fct(active_logits, active_labels) + loss_span

        if self._rl_ratio > 0.0:
            samples_action = Categorical(logits=logits).sample() # [bs, seq]
            samples_start = Categorical(logits=start_dist).sample() # [bs, seq]
            samples_end = Categorical(logits=end_dist).sample() # [bs, seq]
            samples_action_prob = torch.gather(logits, 2, samples_action.unsqueeze(dim=2)) #[bs, seq_len, 1]
            samples_start_prob = torch.gather(start_dist, 2, samples_start.unsqueeze(dim=2)) #[bs, seq_len, 1]
            samples_end_prob = torch.gather(end_dist, 2, samples_end.unsqueeze(dim=2)) #[bs, seq_len, 1]

            samples_action_prob = samples_action_prob.unsqueeze(dim=2)
            samples_start_prob = samples_start_prob.unsqueeze(dim=2)
            samples_end_prob = samples_end_prob.unsqueeze(dim=2)

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
                input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i].tolist())
                sample_str = decode_into_string(input_tokens, samples_action[i].tolist(), samples_start[i].tolist(), samples_end[i].tolist(), samples_mask[i].tolist())
                greedy_str = decode_into_string(input_tokens, greedy_action[i].tolist(), greedy_start[i].tolist(), greedy_end[i].tolist(), samples_mask[i].tolist())
                sample_score = gpt_score(sample_str)
                greedy_score = gpt_score(greedy_str)
                #sample_score = sentence_bleu([input_ref[i].split()], sample_str.split(), weights=weight, smoothing_function=cc.method3)
                #greedy_score = sentence_bleu([input_ref[i].split()], greedy_str.split(), weights=weight, smoothing_function=cc.method3)
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