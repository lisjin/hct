# -*- coding: utf-8 -*
from __future__ import print_function

import sys

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge


class Metrics(object):
    def __init__(self):
        pass

    @staticmethod
    def bleu_score(references, candidates):
        """
        计算bleu值
        :param references: 实际值, list of string
        :param candidates: 验证值, list of string
        :return:
        """

        bleu1s = corpus_bleu([[r] for r in references], candidates, weights=(1.0, 0.0, 0.0, 0.0))
        bleu2s = corpus_bleu([[r] for r in references], candidates, weights=(0.5, 0.5, 0.0, 0.0))
        bleu3s = corpus_bleu([[r] for r in references], candidates, weights=(0.33, 0.33, 0.33, 0.0))
        bleu4s = corpus_bleu([[r] for r in references], candidates, weights=(0.25, 0.25, 0.25, 0.25))
        print("Avg. BLEU(n):\t%.3f (1)\t%.3f (2)\t%.3f (3)\t%.3f (4)" % (bleu1s, bleu2s, bleu3s, bleu4s))
        return (bleu1s, bleu2s, bleu3s, bleu4s)

    @staticmethod
    def em_score(references, candidates):
        total_cnt = len(references)
        match_cnt = 0
        for ref, cand in zip(references, candidates):
            if ref.split() == cand.split():
                match_cnt = match_cnt + 1

        em_score = match_cnt / float(total_cnt)
        print(f'em_score: {em_score:.03f}\tmatch_cnt: {match_cnt}\ttotal_cnt: {total_cnt}')
        return em_score

    @staticmethod
    def rouge_score(references, candidates):
        """
        rouge计算，NLG任务语句生成，词语的recall
        :param references: list string
        :param candidates: list string
        :return:
        """
        rouge = Rouge(metrics=['rouge-n', 'rouge-l'],
                max_n=3,
                apply_avg=True)

        # 遍历计算rouge
        refs = [' '.join(x.split()) for x in references]
        cands = [' '.join(x.split()) for x in candidates]
        scores = rouge.get_scores(cands, refs)

        # 计算平均值
        r1_avg, r2_avg, rl_avg = scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']

        # 输出
        print(f"rouge_1: {r1_avg:.3f}\trouge_2: {r2_avg:.3f}\trouge_l: {rl_avg:.3f}")
        return (r1_avg, r2_avg, rl_avg)


if __name__ == '__main__':
    path_ref = "data/canard/test/sentences.txt"
    path_hypo = "experiments/canard/w_bleu_rl_dot/prediction_task_19_.txt"
    #path_ref = "canard_data/test-tgt.txt"
    #path_hypo = "canard_data/output.tok.txt"
    references = []
    with open(path_ref, "r")as f:
        for line in f:
            ref = line.strip().split("\t")[1]
            references.append(ref.lower())

    candidates = []
    with open(path_hypo, "r")as f:
        for i, line in enumerate(f):
            seq = line.strip().split()
            candidates.append(" ".join(seq).lower())
    # 计算metrics
    Metrics.bleu_score(references, candidates)
    Metrics.em_score(references, candidates)
    Metrics.rouge_score(references, candidates)
