#!/usr/bin/env python3
import argparse
import benepar
import json
import os
import spacy

from nltk import Tree
from tqdm import tqdm

from utils import eprint, _read_leaf, write_lst, concat_path, align_phr_tgt
from utils_data import filter_sources_and_targets


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
    pts_uniq, seen = [], {}
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
    if args.cparse:
        cparse(outs, outs_k, args)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train', choices=['train', 'dev', 'test'])
    ap.add_argument('--data_dir', default='canard')
    ap.add_argument('--cparse', action='store_true')
    args = ap.parse_args()
    main(args)
