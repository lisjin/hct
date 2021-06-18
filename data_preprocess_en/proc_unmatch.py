#!/usr/bin/env python3
import argparse
import json
import os

from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp_models import pretrained
from nltk import Tree
from tqdm import tqdm

from utils import eprint, _read_leaf


def load_examples(unmatch_path, phr_tgt_sps_f, language='en_core_web_sm'):
    """Return list of context, target pairs. If file DNE, write a JSON list of
    target spans corresponding to unmatched phrases.
    """
    tokenizer = SpacyTokenizer(language=language)
    def std_sen(s):
        return [t.text for t in tokenizer.tokenize(s)]

    def align_phr_tgt(phr_spls, tgt_spl):
        sps = []
        for ps in phr_spls:
            pl = len(ps)
            sp = (-1, -1)
            for i in range(len(tgt_spl)):
                if tgt_spl[i:i+pl] == ps:
                    sp = (i, pl)
                    break
            sps.append(sp)
        return sps

    outs = []
    sps_lst = [] if not os.path.exists(phr_tgt_sps_f) else None
    with open(unmatch_path, 'r', encoding='utf8') as f:
        for l in json.load(f).values():
            # Prevent tokenizing [SEP] tag by replacing it with |
            ctx = ' '.join(std_sen(l['src'].split('[CI]')[0].replace('[SEP]', '|')))
            tgt_spl = std_sen(l['tgt'])
            outs.append((ctx, ' '.join(tgt_spl)))

            if sps_lst is not None:
                phr_spls = [std_sen(phr) for phr in l['phr']]
                sps = align_phr_tgt(phr_spls, tgt_spl)
                sps_lst.append(sps[:])
    if sps_lst is not None:
        with open(phr_tgt_sps_f, 'w', encoding='utf8') as f:
            json.dump(sps_lst, f)
    return outs


def lemmatize(outs, lem_f):
    """Lemmatize context, target pairs in `outs`."""
    import stanza
    from stanza.server import CoreNLPClient
    if not os.path.isdir('/home/lisjin/stanza_resources'):
        stanza.download('en')
    if not os.path.isdir('/home/lisjin/stanza_corenlp'):
        stanza.install_corenlp()

    with CoreNLPClient(annotators=['tokenize','ssplit','pos','lemma'], memory=\
        '12G', endpoint='http://localhost:9001', be_quiet=True) as client:
        lems = [' '.join([word.lemma for sentence in client.annotate(s)\
                .sentence for word in sentence.token]) for o in outs for s in o]
        with open(lem_f, 'w', encoding='utf8') as f:
            json.dump(lems, f)


def write_lst(f_name, lst):
    with open(f_name, 'w', encoding='utf8') as f:
        f.writelines(f'{l}\n' for l in lst)


def cparse(outs, ids_f='cpt_ids.txt', pts_f='cpts_uniq.txt', pts_tgt_f='cpts_tgt.txt'):
    pred = pretrained.load_predictor('structured-prediction-constituency-parser')

    def clean_s(s):
        return s.replace('(', '{').replace(')', '}')

    def try_pred(s):
        pt_str = pred.predict(s)['trees']
        try:
            t = Tree.fromstring(pt_str)
            assert(len(s.split()) == len(t.leaves()))
        except:
            pt_str = pred.predict(clean_s(s))['trees']
            t = Tree.fromstring(pt_str, read_leaf=_read_leaf)
            if len(s.split()) != len(t.leaves()):
                t_lvs = ' '.join(t.leaves())
                eprint(f'{s},{t_lvs}')
                return ''
        return pt_str

    pt_ids, pts_uniq = [], []
    seen = {}
    pts_tgt = []
    for i in tqdm(range(len(outs))):
        ids = []
        for s in outs[i][0].split(' | '):
            if s not in seen:
                seen[s] = len(pts_uniq)
                pts_uniq.append(try_pred(s))
            ids.append(seen[s])
        pt_ids.append(','.join(map(str, ids)))

        pts_tgt.append(try_pred(outs[i][1]))

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
        for s in outs[i][0].split(' | '):
            if s not in seen:
                seen[s] = len(pts_uniq)
                pts_uniq.append(pred_lst(s))
            ids.append(seen[s])
        pt_ids.append(','.join(map(str, ids)))

        pts_tgt.append(pred_lst(outs[i][1]))

    assert(len(pt_ids) == len(pts_tgt))
    write_lst(ids_f, pt_ids)
    write_lst(pts_f, pts_uniq)
    write_lst(pts_tgt_f, pts_tgt)


def main(args):
    outs = load_examples(args.unmatch_path, args.phr_tgt_sps_f)
    if args.lem:
        lemmatize(outs, args.lem_f)
    if args.cparse:
        cparse(outs)
    if args.dparse:
        dparse(outs)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--unmatch_path', default='canard/unfound_phrs.json')
    ap.add_argument('--phr_tgt_sps_f', default='canard/phr_tgt_sps.json')
    ap.add_argument('--lem_f', default='canard/unmatch_lems.json')
    ap.add_argument('--lem', action='store_true')
    ap.add_argument('--cparse', action='store_true')
    ap.add_argument('--dparse', action='store_true')
    args = ap.parse_args()
    main(args)
