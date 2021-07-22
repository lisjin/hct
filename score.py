# -*- coding: utf-8 -*
from __future__ import print_function

import sys

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu


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
        https://github.com/pltrdy/rouge
        :param references: list string
        :param candidates: list string
        :return:
        """
        rouge = Rouge()

        # 遍历计算rouge
        rouge1s = []
        rouge2s = []
        rougels = []
        for ref, cand in zip(references, candidates):
            #ref = ' '.join(list(ref))
            #cand = ' '.join(list(cand))
            ref = ' '.join(ref.split())
            cand = ' '.join(cand.split())
            try:
                rouge_score = rouge.get_scores([cand], [ref])
            except:
                print(sys.exc_info()[0])
            rouge_1 = rouge_score[0]["rouge-1"]['f']
            rouge_2 = rouge_score[0]["rouge-2"]['f']
            rouge_l = rouge_score[0]["rouge-l"]['f']

            rouge1s.append(rouge_1)
            rouge2s.append(rouge_2)
            rougels.append(rouge_l)

        # 计算平均值
        rouge1_average = sum(rouge1s) / len(rouge1s)
        rouge2_average = sum(rouge2s) / len(rouge2s)
        rougel_average = sum(rougels) / len(rougels)

        # 输出
        print("rouge_1: %.3f\trouge_2: %.3f\trouge_l: %.3f" \
                % (rouge1_average, rouge2_average, rougel_average))
        return (rouge1_average, rouge2_average, rougel_average)


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
