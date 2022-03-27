import argparse
import json
import os

from allennlp_models import pretrained
from multiprocessing import Pool
from tqdm import tqdm

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def parseargs():
    parser = argparse.ArgumentParser(description="evaluate the related scores")

    parser.add_argument("--ref", type=str, required=True, help="reference")
    parser.add_argument("--hypo", type=str, required=True, help="hypotheis")
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--ref_srl_path', type=str, default='ref_srl.json')

    return parser.parse_args()


def item_helper(item):
    verb_lst = []
    for verb_dct in item['verbs']:
        verb_idx = -1
        s = -1
        spans = []
        n = len(verb_dct['tags'])
        for i, y in enumerate(verb_dct['tags']):
            if y.startswith('B-') or y == 'O':
                if s > -1:
                    spans.append((verb_dct['tags'][s][2:],
                        ' '.join(item['words'][s:i])))
                    s = -1  # close the current span
                if y == 'B-V':
                    verb_idx = item['words'][i]
                    s = -1  # ineligible as start of span
                elif y.startswith('B-'):
                    s = i

            if s > -1 and (i == n - 1 or y.startswith('I-') and\
                    verb_dct['tags'][i + 1] == 'O'):
                spans.append((verb_dct['tags'][s][2:],
                        ' '.join(item['words'][s:i+1])))
                s = -1
        verb_lst.append((verb_idx, spans[:]))
    return verb_lst


def srl_inference(hypo, ref, cuda_device, ref_srl_path):
    pred = pretrained.load_predictor('structured-prediction-srl-bert', cuda_device=cuda_device)
    n_cpu = min(os.cpu_count(), 16)

    def get_spans(batch):
        out_lst = []
        for item in batch:
            out_lst.append(item_helper(item))
        return out_lst

    def helper(lst, bsz=250):
        nonlocal pred, n_cpu
        out_lst = []
        for i in tqdm(range(0, len(lst), bsz)):
            batch = pred.predict_batch_json(lst[i:i+bsz])
            with Pool(n_cpu) as p:
                out_lst.extend(p.map(item_helper, batch))
        assert(len(out_lst) == len(lst))
        return out_lst

    hypo_srl = helper(hypo)
    if os.path.exists(ref_srl_path):
        with open(ref_srl_path) as f:
            ref_srl = json.load(f)
    else:
        ref_srl = helper(ref)
        with open(ref_srl_path, 'w') as f:
            json.dump(ref_srl, f)

    return hypo_srl, ref_srl


def f_score(hypo_srl, ref_srl):

    def get_dict(srl):
        dic_srl = {}
        for tri in srl:
            predicate = tri[0]
            argu = tri[1]
            dic_srl[predicate] = [tuple(x) for x in argu]
        return dic_srl

    predictions, reference, overlap = 0., 0., 0.
    for hypo, ref in zip(hypo_srl, ref_srl):
        hypo_dic = get_dict(hypo)
        ref_dic = get_dict(ref)
        for key in hypo_dic:
            if key in ref_dic:
                common = len(set(hypo_dic[key]) & set(ref_dic[key]))
                overlap += common
            predictions += len(hypo_dic[key])

        reference += sum(len(ref_dic[k]) for k in ref_dic)

    p = 100 * overlap / predictions if predictions > 0. else 0.
    r = 100 * overlap / reference if reference > 0. else 0.
    f_score = 2 * p * r / (p + r) if p + r > 0. else 0.

    print("precision:", p)
    print("recall:", r)
    print("f score:", f_score)

    return f_score


def main(args):
    ref_sen=[]
    with open(args.ref,"r")as file:
        for line in file:
            ref_sen.append({'sentence': line.strip('\n').lower()})

    hypo_sen=[]
    with open(args.hypo, "r")as file:
        for line in file:
            hypo_sen.append({'sentence': line.strip("\n").lower()})

    hypo_srl, ref_srl = srl_inference(hypo_sen, ref_sen, args.cuda_device, args.ref_srl_path)

    f = f_score(hypo_srl, ref_srl)


if __name__ == '__main__':
    parsed_args = parseargs()
    main(parsed_args)
