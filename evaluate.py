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

from transformers.models.gpt2.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer
from nltk.translate.bleu_score import corpus_bleu

from data_loader import DataLoader
from metrics import f1_score, get_entities, classification_report, accuracy_score
from score import Metrics
from sequence_tagger import BertForSequenceTagging
from utils import tags_to_string, get_sp_strs, save_checkpoint, set_logger, Params, RunningAverage


def convert_back_tags(pred_action, pred_start, pred_end, true_action, true_start, true_end, context_lens):
    pred_tags = []
    true_tags = []
    for j in range(len(pred_action)):
        pred_tags.append([])
        true_tags.append([])
        max_i = context_lens[j].item()
        for i in range(len(pred_action[j])):
            if true_action[j][i] == '-1':
                continue

            pstarts, pends = get_sp_strs(pred_start[j][i].tolist(), pred_end[j][i].tolist(), max_i)
            pred_tags[-1].append(f'{pred_action[j][i]}|{pstarts}#{pends}')

            tstarts, tends = get_sp_strs(true_start[j][i].tolist(), true_end[j][i].tolist(), max_i)
            true_tags[-1].append(f'{true_action[j][i]}|{tstarts}#{tends}')
    return pred_tags, true_tags


def eval_to_cpu(out, inp):
    return out.detach().cpu().numpy(), inp.to('cpu').numpy()


def write_pred(ckpt_dir, hypo, epoch, bleu, mode='dev'):
    file_name = os.path.join(ckpt_dir, f"pred_{mode}_{epoch:02d}_{bleu:.4f}.txt")
    with open(file_name, "w") as pred_out:
        pred_out.writelines([f'{hyp}\n' for hyp in hypo])


def evaluate(model, gpt_model, data_iterator, params, epoch, mark='Eval', verbose=False, best_val_bleu=0., optimizer=None, scheduler=None):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()

    idx2tag = params.idx2tag

    true_action_tags = []
    pred_action_tags = []

    true_start_tags = []
    pred_start_tags = []

    true_end_tags = []
    pred_end_tags = []

    # a running average object for loss
    loss_avg = RunningAverage()

    ctx_tokens = []
    src_tokens = []
    references = []
    ctx_lens = []

    for _ in range(params.eval_steps):
        # fetch the next evaluation batch
        batch_data, batch_ref, batch_action, batch_start, batch_end, batch_sp_width, batch_src_idx = next(data_iterator)
        batch_masks = batch_data.gt(0)
        with torch.no_grad():
            output = model((batch_data, batch_ref), gpt_model, attention_mask=batch_masks,
                labels_action=batch_action, labels_start=batch_start, labels_end=batch_end, sp_width=batch_sp_width, src_idx=batch_src_idx)
        loss = output[0]
        loss_avg.update(loss.item())

        ctx_tokens.extend(batch_data)
        src_tokens.extend(batch_src_idx)
        references.extend(batch_ref)
        ctx_lens.extend(batch_masks.long().sum(-1))

        batch_action_output, batch_action = eval_to_cpu(output[1], batch_action)
        batch_start_output, batch_start = eval_to_cpu(output[2], batch_start)
        batch_end_output, batch_end = eval_to_cpu(output[3], batch_end)

        pred_action_tags.extend([[idx2tag.get(idx) for idx in indices] for indices in np.argmax(batch_action_output, axis=2)])
        true_action_tags.extend([[idx2tag.get(idx) if idx != -1 else '-1' for idx in indices] for indices in batch_action])

        pred_start_tags.extend([indices for indices in batch_start_output])
        true_start_tags.extend([indices for indices in batch_start])

        pred_end_tags.extend([indices for indices in batch_end_output])
        true_end_tags.extend([indices for indices in batch_end])

    pred_tags, true_tags = convert_back_tags(pred_action_tags, pred_start_tags,
            pred_end_tags, true_action_tags, true_start_tags, true_end_tags,
            ctx_lens)
    ctx, src = [], []
    for i in range(len(ctx_tokens)):
        ctx.append(model.tokenizer.convert_ids_to_tokens(ctx_tokens[i].tolist()))
        src.append([ctx[-1][x] for x in src_tokens[i]])

    hypo = []
    for i in range(len(pred_tags)):
        ctx_len = ctx_tokens[i].ne(model.pad_token_id).long().sum()
        context = ctx[i][:ctx_len]
        source = src[i][:len(pred_tags[i])]
        pred = tags_to_string(source, pred_tags[i], context=context).strip()
        hypo.append(pred.lower())

    assert len(pred_tags) == len(true_tags)

    for i in range(len(pred_tags)):
        assert len(pred_tags[i]) == len(true_tags[i])

    # logging loss, f1 and report
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
    em_score = Metrics.em_score(references, hypo)
    rouge1, rouge2, rougel = Metrics.rouge_score(references, hypo)
    metrics['bleu1'] = bleu1
    metrics['bleu2'] = bleu2
    metrics['bleu3'] = bleu3
    metrics['bleu4'] = bleu4
    metrics['rouge1'] = rouge1
    metrics['rouge2'] = rouge2
    metrics['rouge-L'] = rougel
    metrics['em_score'] = em_score

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
    parser.add_argument('--model', default='acl19/w_bleu_rl_transfer_token_bugfix', help="Directory containing the trained model")
    parser.add_argument('--gpu', default='0', help="gpu device")
    parser.add_argument('--seed', type=int, default=23, help="random seed for initialization")
    parser.add_argument('--restore_dir', required=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Load the parameters from json file
    json_path = os.path.join(args.model, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Use GPUs if available
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed

    params.batch_size = 1

    # Set the logger
    set_logger(os.path.join(args.model, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Loading the dataset...")

    # Initialize the DataLoader
    data_dir = 'data_preprocess_en/' + args.dataset

    if args.dataset in ["canard_out"]:
        bert_class = params.bert_class # auto
        # bert_class = 'pretrained_bert_models/bert-base-cased/' # manual
    elif args.dataset in ["emnlp19"]:
        bert_class = 'bert-base-chinese' # auto
        # bert_class = 'pretrained_bert_models/bert-base-chinese/' # manual
    elif args.dataset in ["acl19"]:
        bert_class = 'bert-base-chinese' # auto

    data_loader = DataLoader(data_dir, bert_class, params, token_pad_idx=0, tag_pad_idx=-1)

    # Load the model
    model = BertForSequenceTagging.from_pretrained(args.restore_dir, num_labels=len(params.tag2idx))
    model.to(params.device)

    #gpt_model = GPT2LMHeadModel.from_pretrained("./dialogue_model/")
    #gpt_model.to(params.device)
    #gpt_model.eval()
    gpt_model = 'bleu'

    # Load data
    test_data = data_loader.load_data('test')

    # Specify the test set size
    params.test_size = test_data['size']
    params.eval_steps = math.ceil(params.test_size / params.batch_size)
    test_data_iterator = data_loader.data_iterator(test_data, shuffle=False)

    params.tagger_model_dir = args.model

    logging.info("Starting evaluation...")
    epoch = int(os.path.basename(os.path.normpath(args.restore_dir)))
    test_metrics = evaluate(model, gpt_model, test_data_iterator, params, epoch,
            mark='Test')
