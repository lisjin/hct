"""Train and evaluate the model"""
import argparse
import logging
import math
import os
import random
import torch
import utils
import torch.nn as nn

from tqdm import trange
from transformers.optimization import get_linear_schedule_with_warmup, AdamW

from evaluate import evaluate
from data_loader import DataLoader
from sequence_tagger import BertForSequenceTagging
from utils import load_checkpoint, get_config, load_rules

import transformers
transformers.logging.set_verbosity(transformers.logging.ERROR)


def train_epoch(model, data_iterator, optimizer, scheduler, params):
    """Train the model on `steps` batches"""
    model.train()
    loss_avg = utils.RunningAverage()

    scaler = torch.cuda.amp.GradScaler()
    prev_scale = scaler.get_scale()
    one_epoch = trange(params.train_steps)
    for batch in one_epoch:
        batch_data, batch_ref, batch_action, batch_start, batch_end, batch_sp_width, batch_rule, batch_src_idx = next(data_iterator)

        model.zero_grad()
        with torch.cuda.amp.autocast():
            loss = model((batch_data, batch_ref), batch_action,
                    batch_start, batch_end, batch_sp_width, batch_rule, batch_src_idx)[0]

        loss_avg.update(loss.item())
        one_epoch.set_postfix(loss=f'{loss_avg():05.3f}')

        loss /= params.grad_accum_steps
        scaler.scale(loss).backward()
        if (batch + 1) % params.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=params.clip_grad)

            scaler.step(optimizer)
            scaler.update()
            cur_scale = scaler.get_scale()
            if cur_scale >= prev_scale:  # ensure that scaler called optimizer.step
                scheduler.step()
            prev_scale = cur_scale


def train_and_evaluate(model, data_loader, train_data, val_data, optimizer, scheduler, params, model_dir, cur_epoch, best_val_bleu):
    """Train the model and evaluate every epoch."""
    patience_counter = 0
    params.train_steps = math.ceil(params.train_size / params.batch_size)
    params.eval_steps = math.ceil(params.val_size / params.batch_size)
    for epoch in range(cur_epoch, params.epoch_num + 1):
        logging.info("Epoch {}/{}".format(epoch, params.epoch_num))

        train_data_iterator = data_loader.data_iterator(train_data, shuffle=True)
        train_epoch(model, train_data_iterator, optimizer, scheduler, params)

        val_data_iterator = data_loader.data_iterator(val_data, shuffle=False)
        val_metrics = evaluate(model, val_data_iterator, params, epoch, mark='Val', best_val_bleu=best_val_bleu, optimizer=optimizer, scheduler=scheduler)
        if 'best_val_bleu' in val_metrics:
            best_val_bleu = val_metrics['best_val_bleu']
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping and logging best BLEU
        if (patience_counter >= params.patience_num and epoch > params.min_epoch_num) or epoch == params.epoch_num:
            logging.info("Best val BLEU: {:05.2f}".format(best_val_bleu))
            break


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    json_path = os.path.join(args.model, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params.seed = args.seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.rules = load_rules(args.rule_path)
    utils.set_logger(os.path.join(args.model, 'train.log'))

    bert_class = params.bert_class
    data_loader = DataLoader(args.dataset, bert_class, params)
    train_data = data_loader.load_data('train')
    val_data = data_loader.load_data('dev')
    params.train_size = train_data['size']
    params.val_size = val_data['size']

    # Prepare model
    if args.restore_dir is not None:
        logging.info(f'Restoring model from {args.restore_dir}')
        bert_class = args.restore_dir

    config = get_config(data_loader.tokenizer, params, bert_class, args.bleu_rl)
    model = BertForSequenceTagging.from_pretrained(bert_class, config=config)
    model.to(params.device)

    # Prepare optimizer
    if params.full_finetuning:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': params.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
    else: # only finetune the head classifier
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=params.learning_rate, correct_bias=False)
    cur_epoch = 1
    train_steps_per_epoch = math.ceil(params.train_size / params.batch_size)
    num_training_steps = params.epoch_num * train_steps_per_epoch
    num_warmup_steps = train_steps_per_epoch * params.warmup_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, last_epoch=cur_epoch - 2)
    best_val_bleu = 0.
    if args.restore_dir is not None:
        optimizer, scheduler, best_val_bleu = load_checkpoint(optimizer, scheduler, args.restore_dir)
        cur_epoch = int(os.path.basename(os.path.normpath(args.restore_dir))) + 1

    params.tagger_model_dir = args.model
    logging.info("Starting training for {} epoch(s)".format(params.epoch_num - cur_epoch + 1))
    train_and_evaluate(model, data_loader, train_data, val_data, optimizer, scheduler, params, args.model, cur_epoch, best_val_bleu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='acl19', help="Directory containing the dataset")
    parser.add_argument('--rule_path', help='Path to phrase rules for first-level decoder')
    parser.add_argument('--model', default='acl19/w_bleu_rl_transfer_token_bugfix', help="Directory containing the model")
    parser.add_argument('--gpu', default='0', help="gpu device")
    parser.add_argument('--bleu_rl', action='store_true')
    parser.add_argument('--seed', type=int, default=2020, help="random seed for initialization")
    parser.add_argument('--restore_dir', default=None,
                        help="Optional, name of the directory containing weights to reload before training, e.g., 'experiments/conll/'")
    args = parser.parse_args()
    main(args)
