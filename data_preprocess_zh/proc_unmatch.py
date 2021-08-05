#!/usr/bin/env python3
import argparse
import benepar
import json
import os
import spacy

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
    nlp = spacy.load('zh_core_web_md')
    nlp.add_pipe(benepar.BeneparComponent('benepar_zh2'))

    def clean_s(s):
        return s.replace('(', '{').replace(')', '}')

    def try_pred(s):
        try:
            pt_str = list(nlp(s).sents)[0]._.parse_string
            t = Tree.fromstring(pt_str)
            ntok = len(s.split())
            assert(len(t.leaves()) == ntok)
        except BaseException as e:
            print(e.args)
            pt_str = ''
        return pt_str

    pt_ids = []
    pts_uniq, seen = cparse_load(outs, outs_k, args)
    pts_src, pts_tgt = [], []
    for i in tqdm(range(len(outs))):
        ids = []
        for j, s in enumerate(outs[i][0].split(' | ')):
            if s not in seen:
                seen[s] = len(pts_uniq)
                pts_uniq.append(try_pred(s))
            ids.append(seen[s])
        pt_ids.append(','.join(map(str, ids)))

        pts_src.append(try_pred(outs[i][2]))
        pts_tgt.append(try_pred(outs[i][1]))

    assert(len(pt_ids) == len(pts_tgt))
    write_lst(concat_path(args, f'cpt_ids.txt'), pt_ids)
    write_lst(concat_path(args, f'cpts_uniq.txt'), pts_uniq)
    write_lst(concat_path(args, f'cpts_tgt.txt'), pts_tgt)
    write_lst(concat_path(args, 'cpts_src.txt'), pts_src)


def main(args):
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
    ap.add_argument('--tsv_fmt', default='wikisplit')
    ap.add_argument('--dparse', action='store_true')
    args = ap.parse_args()
    main(args)
