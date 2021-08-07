#!/usr/bin/env python3
import argparse
import json
import os

from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp_models import pretrained
from nltk import Tree
from tqdm import tqdm

from utils import eprint, _read_leaf, write_lst, concat_path, align_phr_tgt
from utils_data import yield_sources_and_targets, filter_sources_and_targets


def load_examples(args):
    """Return list of context, target pairs. If file DNE, write a JSON list of
    target spans corresponding to unmatched phrases.
    """
    phr_tgt_sps_f = concat_path(args, 'phr_tgt_sps.json')
    sps_lst = [] if not os.path.exists(phr_tgt_sps_f) else None
    with open(concat_path(args, 'unfound_phrs.json'), 'r', encoding='utf8') as f:
        unmatch_dct = {int(k): v for k, v in json.load(f).items()}
        outs_k = unmatch_dct.keys()
        outs = []
        for k, source, target in filter_sources_and_targets(os.path.join(args.data_dir, f'{args.split}.tsv'), outs_k):
            # Prevent tokenizing [SEP] tag by replacing it with |
            ctx = source.split(' [CI] ')[0].replace('[SEP]', '|')
            outs.append((ctx, target, source))

            if sps_lst is not None:
                sps_lst.append(align_phr_tgt(unmatch_dct[k], target))
    if sps_lst is not None:
        with open(phr_tgt_sps_f, 'w', encoding='utf8') as f:
            json.dump(sps_lst, f)
    return outs_k, outs


def load_all_examples(args):
    unfound_phrs_f = concat_path(args, 'unfound_phrs.json')
    outs_k = []
    if os.path.isfile(unfound_phrs_f):
        with open(unfound_phrs_f, encoding='utf8') as f:
            outs_k = [int(k) for k in json.load(f).keys()]

    outs = []
    for sources, target in yield_sources_and_targets(
        os.path.join(args.data_dir, f'{args.split}.tsv')):
        ctx, src = sources[0].split(' [CI] ')
        ctx = ctx.replace('[SEP]', '|')
        outs.append((ctx, target, src))
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


def cparse_load(outs, outs_k, args):
    pts_uniq = []
    seen = {}
    cpts_uniq_f = concat_path(args, 'cpts_uniq.txt')
    if os.path.isfile(cpts_uniq_f):
        with open(cpts_uniq_f, encoding='utf8') as f:
            pts_uniq = [l.rstrip() for l in f]
        for i in outs_k:
            for s in outs[i][0].split(' | '):
                if s not in seen:
                    seen[s] = len(seen)
        assert(len(seen) == len(pts_uniq))
    return pts_uniq, seen


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

    pos = [(t[0][0].split(' [CI] '), t[1]) for k, t in enumerate(yield_sources_and_targets(os.path.join(args.data_dir, f'{args.split}_pos.tsv')))]
    assert(len(pos) == len(outs))

    pt_ids = []
    pts_uniq, seen = cparse_load(outs, outs_k, args)
    pts_src, pts_tgt = [], []
    for i in tqdm(range(len(outs))):
        ids = []
        pos_cur = pos[i][0][0].split(' [SEP] ')
        for j, s in enumerate(outs[i][0].split(' | ')):
            if s not in seen:
                seen[s] = len(pts_uniq)
                pts_uniq.append(try_pred(s, pos_cur[j]))
            ids.append(seen[s])
        pt_ids.append(','.join(map(str, ids)))

        pts_src.append(try_pred(outs[i][2], pos[i][0][1]))
        pts_tgt.append(try_pred(outs[i][1], pos[i][1]))

    assert(len(pt_ids) == len(pts_tgt))
    all_pref = int(args.load_all)
    write_lst(concat_path(args, f'cpt_ids{all_pref}.txt'), pt_ids)
    write_lst(concat_path(args, f'cpts_uniq{all_pref}.txt'), pts_uniq)
    write_lst(concat_path(args, f'cpts_tgt{all_pref}.txt'), pts_tgt)
    write_lst(concat_path(args, 'cpts_src.txt'), pts_src)


def main(args):
    if args.load_all:  # outs_k contains indices of previously parsed examples
        outs_k, outs = load_all_examples(args)
    else:
        outs_k, outs = load_examples(args)

    if args.lem:
        lemmatize(outs, concat_path(args, 'unmatch_lems.json'))
    if args.cparse:
        cparse(outs, outs_k, args)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train', choices=['train', 'dev', 'test'])
    ap.add_argument('--data_dir', default='canard')
    ap.add_argument('--lem', action='store_true')
    ap.add_argument('--cparse', action='store_true')
    ap.add_argument('--load_all', action='store_true')
    args = ap.parse_args()
    main(args)
