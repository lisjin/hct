"""Evaluate the model"""
import argparse
import logging
import math
import os
import random
import torch
import utils

import numpy as np

from transformers.models.gpt2.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer
from nltk.translate.bleu_score import corpus_bleu

from data_loader import DataLoader
from metrics import f1_score, get_entities, classification_report, accuracy_score
from score import Metrics
from sequence_tagger import BertForSequenceTagging
from utils import tags_to_string

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='acl19', help="Directory containing the dataset")
parser.add_argument('--model', default='acl19/w_bleu_rl_transfer_token_bugfix', help="Directory containing the trained model")
parser.add_argument('--gpu', default='0', help="gpu device")
parser.add_argument('--seed', type=int, default=23, help="random seed for initialization")


def convert_back_tags(pred_action, pred_start, pred_end, true_action, true_start, true_end):
    pred_tags = []
    true_tags = []
    for j in range(len(pred_action)):
        p_tags = []
        t_tags = []
        for i in range(len(pred_action[j])):
            if true_action[j][i] == '-1':
                continue
            p_tag = pred_action[j][i]+"|"+utils.lst2str(pred_start[j][i])+"#"+utils.lst2str(pred_end[j][i])
            p_tags.append(p_tag)
            t_tag = true_action[j][i]+"|"+utils.lst2str(true_start[j][i])+"#"+utils.lst2str(true_end[j][i])
            t_tags.append(t_tag)
        pred_tags.append(p_tags)
        true_tags.append(t_tags)
    return pred_tags, true_tags


def eval_to_cpu(out, inp):
    return out.detach().cpu().numpy(), inp.to('cpu').numpy()


def evaluate(model, gpt_model, data_iterator, params, epoch, mark='Eval', verbose=False):
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
    loss_avg = utils.RunningAverage()

    source_tokens = []
    references = []

    for _ in range(params.eval_steps):
        # fetch the next evaluation batch
        batch_data, batch_token_starts, batch_ref, batch_action, batch_start, batch_end, batch_sp_width = next(data_iterator)
        batch_masks = batch_data.gt(0)
        with torch.no_grad():
            output = model((batch_data, batch_token_starts, batch_ref), gpt_model, attention_mask=batch_masks,
                labels_action=batch_action, labels_start=batch_start, labels_end=batch_end, sp_width=batch_sp_width)
        loss = output[0]
        loss_avg.update(loss.item())

        source_tokens.extend(batch_data)
        references.extend(batch_ref)

        batch_action_output, batch_action = eval_to_cpu(output[1], batch_action)
        batch_start_output, batch_start = eval_to_cpu(output[2], batch_start)
        batch_end_output, batch_end = eval_to_cpu(output[3], batch_end)

        pred_action_tags.extend([[idx2tag.get(idx) for idx in indices] for indices in np.argmax(batch_action_output, axis=2)])
        true_action_tags.extend([[idx2tag.get(idx) if idx != -1 else '-1' for idx in indices] for indices in batch_action])

        pred_start_tags.extend([indices for indices in batch_start_output])
        true_start_tags.extend([indices for indices in batch_start])

        pred_end_tags.extend([indices for indices in batch_end_output])
        true_end_tags.extend([indices for indices in batch_end])

    pred_tags, true_tags = convert_back_tags(pred_action_tags, pred_start_tags, pred_end_tags,
        true_action_tags, true_start_tags, true_end_tags)
    source = []
    for i in range(len(source_tokens)):
        source.append(model.tokenizer.convert_ids_to_tokens(source_tokens[i].tolist()))

    hypo = []
    for i in range(len(pred_tags)):
        src = source[i][:len(pred_tags[i])]
        pred = tags_to_string(src, pred_tags[i]).strip()
        hypo.append(pred.lower())

    if mark == "Test":
        file_name = "/prediction_emnlp"+"_"+str(epoch)+"_.txt"
        pred_out = open(params.tagger_model_dir+file_name, "w")
        for i in range(len(hypo)):
            pred_out.write(hypo[i]+"\n")
        pred_out.close()

    if mark == "Val":
        file_name = "/prediction_acl"+"_"+str(epoch)+"_.txt"
        pred_out = open(params.tagger_model_dir+file_name, "w")
        for i in range(len(hypo)):
            pred_out.write(hypo[i]+"\n")
        pred_out.close()

    assert len(pred_tags) == len(true_tags)

    for i in range(len(pred_tags)):
        assert len(pred_tags[i]) == len(true_tags[i])

    if mark == "Test":
        logging.info("***********EMNLP Test************")

    # logging loss, f1 and report
    metrics = {}
    bleu1, bleu2, bleu3, bleu4 = Metrics.bleu_score(references, hypo)
    em_score = Metrics.em_score(references, hypo)
    rouge1, rouge2, rougel = Metrics.rouge_score(references, hypo)
    metrics['bleu1'] = bleu1*100.0
    metrics['bleu2'] = bleu2*100.0
    metrics['bleu3'] = bleu3*100.0
    metrics['bleu4'] = bleu4*100.0
    metrics['rouge1'] = rouge1*100.0
    metrics['rouge2'] = rouge2*100.0
    metrics['rouge-L'] = rougel*100.0
    metrics['em_score'] = em_score*100.0
    f1 = f1_score(true_tags, pred_tags)
    metrics['loss'] = loss_avg()
    metrics['f1'] = f1
    accuracy = accuracy_score(true_tags, pred_tags)
    metrics['accuracy'] = accuracy
    metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in metrics.items())
    logging.info("- {} metrics: ".format(mark) + metrics_str)

    if verbose:
        report = classification_report(true_tags, pred_tags)
        logging.info(report)
    return metrics


def interAct(model, data_iterator, params, mark='Interactive', verbose=False):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()

    idx2tag = params.idx2tag

    true_tags = []
    pred_tags = []

    # a running average object for loss
    loss_avg = utils.RunningAverage()


    batch_data, batch_token_starts = next(data_iterator)
    batch_masks = batch_data.gt(0)

    batch_output = model((batch_data, batch_token_starts), token_type_ids=None, attention_mask=batch_masks)[0]  # shape: (batch_size, max_len, num_labels)

    batch_output = batch_output.detach().cpu().numpy()

    pred_tags.extend([[idx2tag.get(idx) for idx in indices] for indices in np.argmax(batch_output, axis=2)])

    return(get_entities(pred_tags))


if __name__ == '__main__':
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    tagger_model_dir = 'experiments/' + args.model
    # Load the parameters from json file
    json_path = os.path.join(tagger_model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Use GPUs if available
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed

    params.batch_size = 1

    # Set the logger
    utils.set_logger(os.path.join(tagger_model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Loading the dataset...")

    # Initialize the DataLoader
    data_dir = 'data/' + args.dataset

    if args.dataset in ["canard"]:
        bert_class = 'bert-base-cased' # auto
        # bert_class = 'pretrained_bert_models/bert-base-cased/' # manual
    elif args.dataset in ["emnlp19"]:
        bert_class = 'bert-base-chinese' # auto
        # bert_class = 'pretrained_bert_models/bert-base-chinese/' # manual
    elif args.dataset in ["acl19"]:
        bert_class = 'bert-base-chinese' # auto

    data_loader = DataLoader(data_dir, bert_class, params, token_pad_idx=0, tag_pad_idx=-1)

    # Load the model
    model = BertForSequenceTagging.from_pretrained(tagger_model_dir)
    model.to(params.device)

    #gpt_model = GPT2LMHeadModel.from_pretrained("./dialogue_model/")
    #gpt_model.to(params.device)
    #gpt_model.eval()
    gpt_model = None

    # Load data
    test_data = data_loader.load_data('test')

    # Specify the test set size
    params.test_size = test_data['size']
    params.eval_steps = math.ceil(params.test_size / params.batch_size)
    test_data_iterator = data_loader.data_iterator(test_data, shuffle=False)

    params.tagger_model_dir = tagger_model_dir

    logging.info("- done.")

    logging.info("Starting evaluation...")
    test_metrics = evaluate(model, gpt_model, test_data_iterator, params, 'Test', mark='Test', verbose=True)
