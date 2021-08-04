import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from transformers.models.bert.modeling_bert import BertPreTrainedModel, CrossEntropyLoss
from transformers import BertTokenizer, BertModel
from torch.autograd import Variable
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from single_headed_additive_attn import SingleHeadedAttention
from utils import convert_tokens_to_string, get_sp_strs, safe_log, sample_helper, cls_loss

cc = SmoothingFunction()


class SpanClassifier(nn.Module):
    def __init__(self, hidden_dim, max_relative_position=0., dropout=0.1):
        super(SpanClassifier, self).__init__()
        self.span_st_attn = SingleHeadedAttention(hidden_dim, max_relative_positions=max_relative_position)
        self.span_ed_attn = SingleHeadedAttention(hidden_dim, max_relative_positions=max_relative_position)
        self.src_rule_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.sp_emb = nn.Linear(2 * hidden_dim, hidden_dim)
        self.dropout = dropout
        if max_relative_position > 0.0:
            print("Setting max_relative_position to {}".format(max_relative_position))

    def reset_parameters(self):
        nn.init.normal_(self.sp_emb.weight, std=2e-2)
        nn.init.constant_(self.sp_emb.bias, 0.)
        nn.init.normal_(self.src_rule_proj.weight, std=2e-2)
        nn.init.constant_(self.src_rule_proj.bias, 0.)

    def forward(self, hid, sp_width, max_sp_len, attention_mask, src_hid, src_mask, rule_emb):
        amask = torch.logical_and(src_mask.unsqueeze(2), attention_mask.unsqueeze(1)).float()
        attn_d = amask.sum(-1, keepdim=True)
        attn_d.masked_fill_(attn_d == 0, 1.)
        attn_w0 = amask / attn_d

        sts, eds, masks = [attn_w0], [attn_w0], []
        src_hid = F.relu(self.src_rule_proj(torch.cat((src_hid, rule_emb), 2)), inplace=True)
        hid1, hid2 = src_hid, src_hid
        for i in range(max_sp_len):
            mask = (i < sp_width).float()
            masks.append(mask)
            mask = torch.logical_and(mask.unsqueeze(-1), amask)
            hid1 = self.upd_hid(sts[-1], hid, hid1)
            hid2 = self.upd_hid(eds[-1], hid, hid2)
            sts.append(self.span_st_attn(hid, hid1, mask=mask))
            eds.append(self.span_ed_attn(hid, hid2, mask=mask))
        return torch.stack(sts[1:], -2), torch.stack(eds[1:], -2), torch.stack(masks, -1)

    @staticmethod
    def span_loss(start_dist, end_dist, start_positions, end_positions, seq_masks):
        if not seq_masks.any():
            return 0.
        span_st_loss = cls_loss(start_dist, start_positions, seq_masks)
        span_ed_loss = cls_loss(end_dist, end_positions, seq_masks)
        return span_st_loss + span_ed_loss

    def upd_hid(self, attn_w, hid, hidp):
        hidc = (hid.unsqueeze(1) * attn_w.unsqueeze(-1)).sum(2)
        hidc = F.relu(self.sp_emb(torch.cat((hidp, hidc), 2)), inplace=True)
        hidc = F.dropout(hidc, p=self.dropout, training=self.training)
        return hidc


class BertForSequenceTagging(nn.Module):
    def __init__(self, config):
        super(BertForSequenceTagging, self).__init__()
        self.device = config.device
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_class,
                additional_special_tokens=config.additional_special_tokens)
        self.pad_token_id = config.pad_token_id

        self.tags = config.tags
        self.pad_tag_id = config.pad_tag_id
        self.num_labels = config.num_labels

        self.bert = BertModel.from_pretrained(config.bert_class).to(self.device)
        if config.additional_special_tokens is not None:
            self.bert.resize_token_embeddings(config.vocab_size)
        self.max_sp_len = config.max_sp_len
        self.rules = config.rules
        self.rule_embeds, _ = self.embed_rules_slots(config.additional_special_tokens)
        self.rule_slot_cnts = config.rule_slot_cnts
        self.num_rules = len(self.rules)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.loss_fn = CrossEntropyLoss()
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.rule_classifier = nn.Linear(config.hidden_size, self.num_rules)
        self.span_classifier = SpanClassifier(config.hidden_size)

        self.rl_model = config.rl_model
        self.rl_ratio = config.rl_ratio
        self.bleu_fn = partial(sentence_bleu, weights=(.25,) * 4,
                smoothing_function=cc.method3)

    def unpack_data(self, input_data, src_idx):
        input_ids, input_ref = input_data
        attention_mask = input_ids.ne(self.pad_token_id)
        seq_output = self.bert(input_ids, attention_mask=attention_mask)[0]
        src_mask = src_idx.ne(self.pad_token_id)
        src_output = self.gather_embed(seq_output, src_idx, src_mask)
        src_idx = input_ids.gather(1, src_idx) * src_mask.long()
        return input_ids, input_ref, attention_mask, seq_output, src_mask,\
                src_output, src_idx

    def pred_tags(self, src_mask, src_output, src_idx, labels_action, rule,
            decode=False):
        logits = self.classifier(src_output)
        act_loss_mask = labels_action.ne(self.pad_tag_id)
        loss_action = self.get_active_loss(act_loss_mask, logits, labels_action,
                self.num_labels)
        rule_logits = self.rule_classifier(src_output)
        loss_rule = self.get_active_loss(act_loss_mask, rule_logits, rule,
                self.num_rules)
        if not self.training:
            rule = rule_logits.argmax(2)
        rule_emb = self.rule_embeds[rule]
        return logits, act_loss_mask, loss_action, rule_logits, loss_rule, rule_emb

    def pred_spans(self, labels_start, labels_end, seq_output, sp_width,
            attention_mask, src_output, src_mask, rule, rule_emb):
        max_sp_len = labels_start.shape[-1]
        start_dist, end_dist, sp_loss_mask = self.span_classifier(seq_output,
                sp_width, max_sp_len, attention_mask, src_output, src_mask, rule_emb)
        start_outputs = start_dist.argmax(dim=-1)
        end_outputs = end_dist.argmax(dim=-1)

        sp_loss_mask = torch.logical_and(rule.gt(0).unsqueeze(2), sp_loss_mask).float()
        loss_span = self.span_classifier.span_loss(start_dist, end_dist,
                labels_start, labels_end, sp_loss_mask)
        return max_sp_len, start_dist, end_dist, sp_loss_mask, start_outputs, end_outputs, loss_span

    def apply_rl(self, logits, rule_logits, start_dist, end_dist, start_outputs,
            end_outputs, src_idx, ctx_idx, attention_mask, act_loss_mask,
            sp_loss_mask, input_ref, max_sp_len, perm=(0, 2, 1, 3)):
        samples_action, samples_action_prob, greedy_action = self.get_sample_greedy(logits)
        samples_rule, samples_rule_prob, greedy_rule = self.get_sample_greedy(rule_logits)

        bsz, seq_len, _, full_len = start_dist.shape
        start_dist, samples_start, samples_start_prob = self.sample_sp(start_dist, perm, seq_len, full_len)
        end_dist, samples_end, samples_end_prob = self.sample_sp(end_dist, perm, seq_len, full_len)

        nview = (bsz, max_sp_len, seq_len)
        samples_start, samples_end, samples_start_prob, samples_end_prob =\
                map(lambda x: self.reshape_sp(x, nview, perm[:-1]),
                    (samples_start, samples_end, samples_start_prob, samples_end_prob))

        min_r, max_r = 1., -1.
        rewards = []
        act_loss_mask = act_loss_mask.float()
        for i in range(len(samples_start)):
            src_len, src_tokens = self.get_len_tokens(act_loss_mask[i], src_idx[i])
            ctx_len, ctx_tokens = self.get_len_tokens(attention_mask[i], ctx_idx[i])

            sample_str = self.decode_into_string(src_tokens, samples_action[i].tolist(), samples_rule[i].tolist(), samples_start[i].tolist(), samples_end[i].tolist(), src_len, context=ctx_tokens, context_len=ctx_len)
            greedy_str = self.decode_into_string(src_tokens, greedy_action[i].tolist(), greedy_rule[i].tolist(), start_outputs[i].tolist(), end_outputs[i].tolist(), src_len, context=ctx_tokens, context_len=ctx_len)

            input_ref_lst = [input_ref[i].split()]
            sample_score = self.bleu_fn(input_ref_lst, sample_str.split())
            greedy_score = self.bleu_fn(input_ref_lst, greedy_str.split())
            rewards.append(sample_score - greedy_score)
            min_r = min(min_r, rewards[-1])
            max_r = max(max_r, rewards[-1])

        rewards = torch.as_tensor(rewards, device=logits.device).unsqueeze(1)
        rewards = (rewards - min_r) / (max_r - min_r)
        loss_action_rl = self.rl_loss(samples_action_prob.squeeze(2), act_loss_mask, rewards)
        loss_rule_rl = self.rl_loss(samples_rule_prob.squeeze(2), act_loss_mask, rewards)

        rewards = rewards.unsqueeze(2)
        loss_st_rl = self.rl_loss(samples_start_prob, sp_loss_mask, rewards)
        loss_ed_rl = self.rl_loss(samples_end_prob, sp_loss_mask, rewards)

        loss_rl = loss_action_rl + loss_rule_rl + loss_st_rl + loss_ed_rl
        return loss_rl

    def get_active_loss(self, loss_mask, logits, labels, num_labels):
        active_loss = loss_mask.view(-1)
        active_logits = logits.view(-1, num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        return self.loss_fn(active_logits, active_labels)

    def forward(self, input_data, labels_action, labels_start, labels_end,
            sp_width, rule, src_idx):
        input_ids, input_ref, attention_mask, seq_output, src_mask, src_output,\
                src_idx = self.unpack_data(input_data, src_idx)
        logits, act_loss_mask, loss_action, rule_logits, loss_rule, rule_emb =\
                self.pred_tags(src_mask, src_output, src_idx, labels_action, rule)
        max_sp_len, start_dist, end_dist, sp_loss_mask, start_outputs,\
                end_outputs, loss_span = self.pred_spans(labels_start, labels_end,
                seq_output, sp_width, attention_mask, src_output, src_mask,
                rule, rule_emb)
        loss = loss_action + loss_rule + loss_span

        if self.rl_model is not None:
            loss_rl = self.apply_rl(logits, rule_logits, start_dist, end_dist,
                    start_outputs, end_outputs, src_idx, input_ids, attention_mask,
                    act_loss_mask, sp_loss_mask, input_ref, max_sp_len)
            loss = (1. - self.rl_ratio) * loss + self.rl_ratio * loss_rl

        outputs = (loss, logits, rule_logits, start_outputs, end_outputs)
        return outputs

    @staticmethod
    def get_sample_greedy(logits):
        samples, samples_prob = sample_helper(logits)
        greedy = logits.argmax(dim=-1)
        return samples, samples_prob, greedy

    @staticmethod
    def sample_sp(logits, perm, seq_len, full_len):
        logits = logits.permute(*perm).reshape(-1, seq_len, full_len)
        samples, samples_prob = sample_helper(logits)
        return logits, samples, samples_prob

    @staticmethod
    def reshape_sp(inp, nview, perm):
        return inp.view(*nview).permute(*perm)

    @staticmethod
    def rl_loss(prob, mask, rewards):
        loss = -F.log_softmax(prob + safe_log(mask), -1)
        loss *= rewards * mask
        return loss.sum() / mask.sum()

    @staticmethod
    def gather_embed(inp_emb, inp_idx, inp_mask, dim=1):
        expand_dims = (-1,) * (dim + 1)
        ret = inp_emb.gather(dim, inp_idx.unsqueeze(2).expand(*expand_dims,
            inp_emb.shape[-1])) * inp_mask.unsqueeze(2).to(inp_emb.dtype)
        return ret

    def embed_rules_slots(self, additional_special_tokens, fmask='{}'):
        # TODO: test whether slot embeddings are helpful
        ret, ret_ids = [], []
        mlen = 0
        for rule in self.rules:
            slot_id = 0
            ret.append([])
            ret_ids.append([])
            ret[-1].append(self.tokenizer.cls_token_id)
            j = len(ret[-1])
            for token in rule.split():
                if token == fmask:
                    ret_cur = [additional_special_tokens[slot_id]]
                    slot_id += 1
                    ret_ids[-1].append(j)
                else:
                    ret_cur = self.tokenizer.tokenize(token)
                ret[-1].extend(self.tokenizer.convert_tokens_to_ids(ret_cur))
                j += len(ret_cur)
            mlen = max(mlen, len(ret[-1]))
        for i, x in enumerate(ret):
            ret[i].extend([self.tokenizer.pad_token_id] * (mlen - len(x)))
            ret_ids[i].extend([0] * (self.max_sp_len - len(ret_ids[i])))
        ret = torch.tensor(ret, dtype=torch.long, device=self.device)
        ret_ids = torch.tensor(ret_ids, dtype=torch.long, device=self.device)

        ret = self.bert(ret, attention_mask=ret.ne(self.tokenizer.pad_token_id))[0]
        return Variable(ret[:, 0]), Variable(self.gather_embed(ret, ret_ids, ret_ids.ne(0)))

    def get_len_tokens(self, loss_mask, inp_idx):
        inp_len = loss_mask.long().sum()
        inp_tokens = self.tokenizer.convert_ids_to_tokens(inp_idx[:inp_len].tolist())
        return inp_len, inp_tokens

    def get_labels(self, label_action, label_rule, label_start, label_end, src_len, context_len, null_i=0):
        labels = []
        for idx in range(src_len):
            st, ed = null_i, null_i
            action = self.tags[label_action[idx]]
            rule = label_rule[idx]
            if rule > 0:
                st, ed = get_sp_strs(label_start[idx], label_end[idx], context_len)
                rule = label_rule[idx]
            labels.append(f'{action}|{st}#{ed}|{rule}')
        return labels

    def tags_to_string(self, source, labels, context=None, ignore_toks=set(['[SEP]', '[CLS]', '[UNK]', '|', '*'])):
        output_tokens = []
        for token, tag in zip(source, labels):
            action, added_phrase, rule_id = tag.split('|')
            rule_id = int(rule_id)
            slot_cnt = self.rule_slot_cnts[rule_id]
            starts, ends = added_phrase.split("#")
            starts, ends = map(lambda x: x.split(','), (starts, ends))
            sub_phrs = []
            for i, start in enumerate(starts):
                s_i, e_i = int(start), int(ends[i])
                add_phrase = ' '.join([s for s in context[s_i:e_i+1] if s not in ignore_toks])
                if add_phrase:
                    sub_phrs.append(add_phrase)
                    if len(sub_phrs) == slot_cnt:
                        break
            sub_phrs.extend([''] * (slot_cnt - len(sub_phrs)))
            phr_toks = self.rules[rule_id].format(*sub_phrs).strip().split()
            output_tokens.extend(phr_toks)
            if action == 'KEEP':
                if token not in ignore_toks:
                    output_tokens.append(token)
            if len(output_tokens) > len(context):
                break

        if not output_tokens:
           output_tokens.append('*')
        elif len(output_tokens) > 1 and output_tokens[-1] == '*':
           output_tokens = output_tokens[:-1]
        return convert_tokens_to_string(output_tokens)

    def decode_into_string(self, source, label_action, label_rule, label_start, label_end, src_len, context=None, context_len=0):
        if context is None:
            context = source
            context_len = src_len
        context = context[:context_len]
        labels = self.get_labels(label_action, label_rule, label_start, label_end, src_len, context_len)
        out_str = self.tags_to_string(source, labels, context=context)
        return out_str
