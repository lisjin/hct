import argparse
import json
import os

from score import Metrics


def main(args):
    with open(f'data_preprocess_{args.lang}/{args.dataset}_out/test/sentences{args.f_suf}.txt', 'r') as f:
        tgts0 = [' '.join(l.strip().split('\t')[-1].split()) for l in f]

    ckpt_path = f'experiments/{args.dataset}21_08-15{args.domain_suf}{args.f_suf}/{args.epoch}'
    test_pred = [x for x in os.listdir(ckpt_path) if x.startswith('pred_test')][0]
    print(f'Evaluating from {test_pred}')
    with open(os.path.join(ckpt_path, test_pred)) as f:
        hyps0 = [' '.join(l.strip().split()) for l in f]

    with open(f'data_preprocess_{args.lang}/{args.dataset}/domain_rng.json') as f:
        domain_rng = json.load(f)
    for k, v in domain_rng['test'].items():
        print(k)
        tgts, hyps = tgts0[v[0]:v[1]], hyps0[v[0]:v[1]]
        bleu1, bleu2, bleu3, bleu4 = Metrics.bleu_score(tgts, hyps)
        rouge1, rouge2, rougel = Metrics.rouge_score(tgts, hyps)
        em_score = Metrics.em_score(tgts, hyps)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='canard')
    ap.add_argument('--lang', default='en')
    ap.add_argument('--domain_suf', default='')
    ap.add_argument('--f_suf', default='')
    ap.add_argument('--epoch', required=True)
    args = ap.parse_args()
    main(args)
