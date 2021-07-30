"""Evaluate the model"""
import argparse
import logging
import math
import numpy as np
import os
import re
import random
import shutil
import torch

from transformers import BertTokenizer
from nltk.translate.bleu_score import corpus_bleu

from data_loader import DataLoader
from metrics import f1_score, get_entities, classification_report, accuracy_score
from score import Metrics
from sequence_tagger import BertForSequenceTagging
from utils import tags_to_string, get_sp_strs, save_checkpoint, set_logger, Params, RunningAverage, load_rules


def convert_back_tags(pred_action, pred_rule, pred_start, pred_end, true_action, true_rule, true_start, true_end, context_lens):
    pred_tags = []
    true_tags = []
    for i in range(len(pred_action)):
        pred_tags.append([])
        true_tags.append([])
        for j in range(len(pred_action[i])):
            pstarts, pends = get_sp_strs(pred_start[i][j].tolist(), pred_end[i][j].tolist(), context_lens[j])
            pred_tags[-1].append(f'{pred_action[i][j]}|{pstarts}#{pends}|{pred_rule[i][j]}')

            tstarts, tends = get_sp_strs(true_start[i][j].tolist(), true_end[i][j].tolist(), context_lens[j])
            true_tags[-1].append(f'{true_action[i][j]}|{tstarts}#{tends}|{true_rule[i][j]}')
    return pred_tags, true_tags


def eval_to_cpu(out, inp):
    return out.detach().cpu().numpy(), inp.to('cpu').numpy()


def write_pred(ckpt_dir, hypo, epoch, bleu, mode='dev'):
    file_name = os.path.join(ckpt_dir, f"pred_{mode}_{epoch:02d}_{bleu:.4f}.txt")
    with open(file_name, "w") as pred_out:
        pred_out.writelines([f'{hyp}\n' for hyp in hypo])


def evaluate(model, data_iterator, params, epoch, mark='Eval', verbose=False, best_val_bleu=0., optimizer=None, scheduler=None):
    """Evaluate the model on `steps` batches."""
    model.eval()

    idx2tag = params.idx2tag
    true_action_tags, pred_action_tags = [], []
    true_rule_tags, pred_rule_tags = [], []
    true_start_tags, pred_start_tags = [], []
    true_end_tags, pred_end_tags = [], []
    ctx_tokens, ctx_lens = [], []
    src_tokens, src_lens = [], []
    references = []
    loss_avg = RunningAverage()

    for _ in range(params.eval_steps):
        batch_data, batch_ref, batch_action, batch_start, batch_end, batch_sp_width, batch_rule, batch_src_idx = next(data_iterator)
        with torch.no_grad():
            loss, logits, rule_logits, start_logits, end_logits = model((batch_data, batch_ref), batch_action, batch_start,
                    batch_end, batch_sp_width, batch_rule, batch_src_idx)
        loss_avg.update(loss.item())

        ctx_tokens.extend(batch_data)
        src_tokens.extend(batch_src_idx)
        references.extend(batch_ref)
        src_lens.extend(batch_src_idx.ne(model.pad_token_id).long().sum(-1).tolist())
        ctx_lens.extend(batch_data.ne(model.pad_token_id).long().sum(-1).tolist())

        batch_action_output, batch_action = eval_to_cpu(logits, batch_action)
        batch_rule_output, batch_rule = eval_to_cpu(rule_logits, batch_rule)
        batch_start_output, batch_start = eval_to_cpu(start_logits, batch_start)
        batch_end_output, batch_end = eval_to_cpu(end_logits, batch_end)

        batch_action_output = np.argmax(batch_action_output, axis=2)
        batch_rule_output = np.argmax(batch_rule_output, axis=2)
        for i, cur_len in enumerate(src_lens[-batch_action.shape[0]:]):
            pred_action_tags.append([idx2tag[x] for x in batch_action_output[i, :cur_len]])
            true_action_tags.append([idx2tag[x] for x in batch_action[i, :cur_len]])
            pred_rule_tags.append(batch_rule_output[i, :cur_len])
            true_rule_tags.append(batch_rule[i, :cur_len])
            pred_start_tags.append(batch_start_output[i, :cur_len])
            true_start_tags.append(batch_start[i, :cur_len])
            pred_end_tags.append(batch_end_output[i, :cur_len])
            true_end_tags.append(batch_end[i, :cur_len])

    pred_tags, true_tags = convert_back_tags(pred_action_tags, pred_rule_tags,
            pred_start_tags, pred_end_tags, true_action_tags, true_rule_tags,
            true_start_tags, true_end_tags, ctx_lens)
    assert len(pred_tags) == len(true_tags)

    ctx, src = [], []
    for i in range(len(ctx_tokens)):
        ctx.append(model.tokenizer.convert_ids_to_tokens(ctx_tokens[i].tolist()))
        src.append([ctx[-1][x] for x in src_tokens[i]])

    hypo = []
    for i, src_len in enumerate(src_lens):
        context = ctx[i][:ctx_lens[i]]
        source = src[i][:src_len]
        pred = tags_to_string(source, pred_tags[i], model.rules, model.rule_slot_cnts, context=context).strip()
        hypo.append(pred.lower())

    for i in range(len(pred_tags)):
        assert len(pred_tags[i]) == len(true_tags[i])

    metrics = {}
    bleu1, bleu2, bleu3, bleu4 = Metrics.bleu_score(references, hypo)
    ckpt_dir = os.path.join(params.tagger_model_dir, f'{epoch:02d}')
    if mark == "Test":
        write_pred(ckpt_dir, hypo, epoch, bleu4, mode='test')
    elif mark == "Val":
        if bleu4 > best_val_bleu:
            ckpt_files = [os.path.join(params.tagger_model_dir, x) for x in\
                    os.listdir(params.tagger_model_dir) if re.search(r'^\d+$', x)]
            if len(ckpt_files) > 2:  # keep only most recent top-3
                shutil.rmtree(min(ckpt_files, key=os.path.getctime))
            save_checkpoint(model, ckpt_dir, optimizer, scheduler)
            write_pred(ckpt_dir, hypo, epoch, bleu4)
            metrics['best_val_bleu'] = bleu4
    metrics['em_score'] = Metrics.em_score(references, hypo)
    metrics['rouge1'], metrics['rouge2'], metrics['rouge-L'] = Metrics.rouge_score(references, hypo)
    metrics['bleu1'], metrics['bleu2'], metrics['bleu3'], metrics['bleu4'] = bleu1, bleu2, bleu3, bleu4

    metrics['loss'] = loss_avg()
    metrics['f1'] = f1_score(true_tags, pred_tags)
    metrics['accuracy'] = accuracy_score(true_tags, pred_tags)
    logging.info('\t'.join(f"{k}: {metrics[k]:.3f}" for k in ['loss', 'f1', 'accuracy']))

    if verbose:
        report = classification_report(true_tags, pred_tags)
        logging.info(report)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='acl19', help="Directory containing the dataset")
    parser.add_argument('--rule_path', help='Path to phrase rules for first-level decoder')
    parser.add_argument('--model', default='acl19/w_bleu_rl_transfer_token_bugfix', help="Directory containing the trained model")
    parser.add_argument('--gpu', default='0', help="gpu device")
    parser.add_argument('--seed', type=int, default=23, help="random seed for initialization")
    parser.add_argument('--restore_dir', required=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    json_path = os.path.join(args.model, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed
    params.batch_size = 1
    bert_class = params.bert_class
    params.rules, params.rule_slot_cnts = load_rules(args.rule_path)

    set_logger(os.path.join(args.model, 'evaluate.log'))
    logging.info("Loading the dataset...")

    config = get_config(data_loader.tokenizer, params, args.restore_dir, args.bleu_rl)
    model = BertForSequenceTagging.from_pretrained(bert_class, config=config)
    if args.bleu_rl:
        model._rl_model = 'bleu'
    model.to(params.device)

    data_loader = DataLoader(args.dataset, bert_class, params)
    test_data = data_loader.load_data('test')
    params.test_size = test_data['size']
    params.eval_steps = math.ceil(params.test_size / params.batch_size)
    params.tagger_model_dir = args.model

    logging.info("Starting evaluation...")
    test_data_iterator = data_loader.data_iterator(test_data, shuffle=False)
    epoch = int(os.path.basename(os.path.normpath(args.restore_dir)))
    test_metrics = evaluate(model, test_data_iterator, params, epoch,
            mark='Test')
