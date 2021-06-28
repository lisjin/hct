#!/usr/bin/env python3
import argparse
import json
import os

from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp_models import pretrained
from nltk import Tree
from tqdm import tqdm

from utils import eprint, _read_leaf, write_lst, concat_path
from utils_data import yield_sources_and_targets


def load_examples(args):
    """Return list of context, target pairs. If file DNE, write a JSON list of
    target spans corresponding to unmatched phrases.
    """

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

    outs_k, outs = set(), []
    phr_tgt_sps_f = concat_path(args, 'phr_tgt_sps.json')
    sps_lst = [] if not os.path.exists(phr_tgt_sps_f) else None
    with open(concat_path(args, 'unfound_phrs.json'), 'r', encoding='utf8') as f:
        unmatch_dct = json.load(f)
        for k, l in unmatch_dct.items():
            # Prevent tokenizing [SEP] tag by replacing it with |
            ctx = l['src'].split(' [CI] ')[0].replace('[SEP]', '|')
            tgt_spl = l['tgt'].split()
            outs.append((ctx, l['tgt']))
            outs_k.add(int(k))

            if sps_lst is not None:
                phr_spls = [phr.split() for phr in l['phr']]
                sps = align_phr_tgt(phr_spls, tgt_spl)
                sps_lst.append(sps[:])
    if sps_lst is not None:
        with open(phr_tgt_sps_f, 'w', encoding='utf8') as f:
            json.dump(sps_lst, f)
    return outs_k, outs


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


def cparse(outs, outs_k, args):
    pred = pretrained.load_predictor('structured-prediction-constituency-parser')

    def clean_s(s):
        return s.replace('(', '{').replace(')', '}')

    def try_pred(s, p):
        try:
            pt_str = pred.predict_w_pos(s, p)['trees']
            ntok = len(s.split())
        except AssertionError:
            return ''
        try:
            t = Tree.fromstring(pt_str)
            assert(len(t.leaves()) == ntok)
        except:
            pt_str = pred.predict_w_pos(clean_s(s), p)['trees']
            t = Tree.fromstring(pt_str, read_leaf=_read_leaf)
            if len(t.leaves()) != ntok:
                t_lvs = ' '.join(t.leaves())
                eprint(f'{s},{t_lvs}')
                return ''
        return pt_str

    pos = [(t[0][0].split(' [CI] ')[0], t[1]) for k, t in enumerate(yield_sources_and_targets(os.path.join(args.data_dir, f'{args.split}_pos.tsv'), args.tsv_fmt)) if k in outs_k]
    assert(len(pos) == len(outs))

    pt_ids, pts_uniq = [], []
    seen = {}
    pts_tgt = []
    for i in tqdm(range(len(outs))):
        ids = []
        pos_cur = pos[i][0].split(' [SEP] ')
        for j, s in enumerate(outs[i][0].split(' | ')):
            if s not in seen:
                seen[s] = len(pts_uniq)
                pts_uniq.append(try_pred(s, pos_cur[j]))
            ids.append(seen[s])
        pt_ids.append(','.join(map(str, ids)))

        pts_tgt.append(try_pred(outs[i][1], pos[i][1]))

    assert(len(pt_ids) == len(pts_tgt))
    write_lst(concat_path(args, 'cpt_ids.txt'), pt_ids)
    write_lst(concat_path(args, 'cpts_uniq.txt'), pts_uniq)
    write_lst(concat_path(args, 'cpts_tgt.txt'), pts_tgt)


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
    outs_k, outs = load_examples(args)
    if args.lem:
        lemmatize(outs, concat_path(args, 'unmatch_lems.json'))
    if args.cparse:
        cparse(outs, outs_k, args)
    if args.dparse:
        dparse(outs)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train', choices=['train', 'dev', 'test'])
    ap.add_argument('--data_dir', default='canard')
    ap.add_argument('--lem', action='store_true')
    ap.add_argument('--cparse', action='store_true')
    ap.add_argument('--tsv_fmt', default='wikisplit')
    ap.add_argument('--dparse', action='store_true')
    args = ap.parse_args()
    main(args)
