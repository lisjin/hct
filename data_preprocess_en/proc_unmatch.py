#!/usr/bin/env python3
import argparse
import json

from allennlp_models import pretrained
from collections import Counter
from nltk import Tree
from tqdm import tqdm


def load_examples(unmatch_path):
    """Return list of unmatched examples and print frequency of unmatched
    phrases containing common prepositions.
    """
    outs = []
    with open(unmatch_path, 'r', encoding='utf8') as f:
        for l in json.load(f).values():
            ctx = l['src'].split('[CI]')[0]
            outs.extend([(phr, ctx, l['tgt']) for phr in l['phr']])
    return outs


def lemmatize(outs):
    """Lemmatize unmatched phrase, context, and target triples in `outs`."""
    import stanza
    from stanza.server import CoreNLPClient
    stanza.download('en')
    stanza.install_corenlp()

    with CoreNLPClient(annotators=['tokenize','ssplit','pos','lemma'], memory=\
        '12G', endpoint='http://localhost:9001', be_quiet=True) as client:
        # Prevent tokenizing [SEP] tag by replacing it with |
        lems = [' '.join([word.lemma for sentence in client.annotate(s.replace(
            '[SEP]', '|')).sentence for word in sentence.token]).replace('|',
                '[SEP]') for o in outs for s in o]
        with open('unmatch_lems.json', 'w', encoding='utf8') as f:
            json.dump(lems, f)


def write_lst(f_name, lst):
    with open(f_name, 'w', encoding='utf8') as f:
        f.writelines(f'{l}\n' for l in lst)


def cparse(outs, ids_f='cpt_ids.txt', pts_f='cpts_uniq.txt', pts_tgt_f='cpts_tgt.txt'):
    pred = pretrained.load_predictor('structured-prediction-constituency-parser')

    def clean_s(s):
        return s.replace('(', '-LRB-').replace(')', '-RRB-')

    def try_pred(s):
        pt_str = pred.predict(s)['trees']
        try:
            Tree.fromstring(pt_str)
        except ValueError:
            pt_str = pred.predict(clean_s(s))['trees']
        return pt_str

    pt_ids, pts_uniq = [], []
    seen = {}
    pts_tgt = []
    for i in tqdm(range(len(outs))):
        ids = []
        for s in outs[i][1].split('[SEP]'):
            if s not in seen:
                seen[s] = len(pts_uniq)
                pts_uniq.append(try_pred(s))
            ids.append(seen[s])
        pt_ids.append(','.join(map(str, ids)))

        pts_tgt.append(try_pred(outs[i][2]))

    assert(len(pt_ids) == len(pts_tgt))
    write_lst(ids_f, pt_ids)
    write_lst(pts_f, pts_uniq)
    write_lst(pts_tgt_f, pts_tgt)


def dparse(outs, ids_f='dpt_ids.txt', pts_f='dpts_uniq.txt', pts_tgt_f='dpts_tgt.txt'):
    pred = pretrained.load_predictor('structured-prediction-biaffine-parser')

    def pred_lst(s, keys=('pos', 'predicted_dependencies', 'predicted_heads')):
        dct = pred.predict(s)
        return ' '.join([str(x) for k in keys for x in dct[k]])

    pt_ids, pts_uniq = [], []
    seen = {}
    pts_tgt = []
    for i in tqdm(range(len(outs))):
        ids = []
        for s in outs[i][1].split('[SEP]'):
            if s not in seen:
                seen[s] = len(pts_uniq)
                pts_uniq.append(pred_lst(s))
            ids.append(seen[s])
        pt_ids.append(','.join(map(str, ids)))

        pts_tgt.append(pred_lst(outs[i][2]))

    assert(len(pt_ids) == len(pts_tgt))
    write_lst(ids_f, pt_ids)
    write_lst(pts_f, pts_uniq)
    write_lst(pts_tgt_f, pts_tgt)


def main(args):
    outs = load_examples(args.unmatch_path)
    if args.lem:
        lemmatize(outs)
    if args.cparse:
        cparse(outs)
    if args.dparse:
        dparse(outs)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--unmatch_path', default='canard_out/unfound_phrs.json')
    ap.add_argument('--lem', action='store_true')
    ap.add_argument('--cparse', action='store_true')
    ap.add_argument('--dparse', action='store_true')
    args = ap.parse_args()
    main(args)
