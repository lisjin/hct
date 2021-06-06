#!/usr/bin/env python3
import argparse
import json

from allennlp_models import pretrained
from collections import Counter

PREP = ['besides', 'aside from', 'in addition to', 'other than', 'along with',
    'in regards to', 'regarding']


def prepos_freq(unmatch_path):
    """Return list of unmatched examples and print frequency of unmatched
    phrases containing common prepositions.
    """
    prep_cnt = Counter(PREP)
    outs = []
    cur = []
    with open(unmatch_path, 'r', encoding='utf8') as f:
        for i, line in enumerate(f.readlines()):
            c = i % 4
            if c == 0 or c == 3:
                continue
            elif c == 1:  # unmatched phrase
                s = line.rstrip()[17:]
                cur.append(s)
                for p in prep_cnt:
                    if s.find(p) > -1:
                        prep_cnt[p] += 1
                        break
            elif c == 2:  # context
                s = line.rstrip()[8:].split('[ci]')[0]
                cur.append(s)
                assert(len(cur) == 2)
                outs.append(cur[:])
                cur.clear()

    for k, v in prep_cnt.items():
        print(f'{k}:\t{(v / float(len(outs))):.4f}')
    return outs


def lemmatize(outs):
    """Lemmatize unmatched phrase and context pairs in `outs`."""
    import stanza
    from stanza.server import CoreNLPClient
    stanza.download('en')
    stanza.install_corenlp()

    with CoreNLPClient(annotators=['tokenize','ssplit','pos','lemma'], memory=\
        '12G', endpoint='http://localhost:9001', be_quiet=True) as client:
        # Prevent tokenizing [sep] tag by replacing it with |
        lems = [' '.join([word.lemma for sentence in client.annotate(s.replace(
            '[sep]', '|')).sentence for word in sentence.token]).replace('|',
                '[sep]') for o in outs for s in o]
        with open('unmatch_lems.json', 'w', encoding='utf8') as f:
            json.dump(lems, f)


def cparse(outs, ids_f='pt_ids.txt', pts_f='pts_uniq.txt'):
    pred = pretrained.load_predictor('structured-prediction-constituency-parser')
    seen = {}
    pt_ids = []
    pts_uniq = []
    for o in outs:
        ids = []
        for s in o[1].split('[sep]'):
            if s not in seen:
                pt = pred.predict(s)
                seen[s] = len(pts_uniq)
                pts_uniq.append(pt['trees'])
            ids.append(seen[s])
    pt_ids.append(','.join(map(str, ids)))

    with open(ids_f, 'w', encoding='utf8') as f:
        f.writelines(f'{l}\n' for l in pt_ids)
    with open(pts_f, 'w', encoding='utf8') as f:
        f.writelines(f'{l}\n' for l in pts_uniq)


def main(args):
    outs = prepos_freq(args.unmatch_path)
    if args.lem:
        lemmatize(outs)
    if args.cparse:
        cparse(outs)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--unmatch_path', default='canard_unmatch.txt')
    ap.add_argument('--lem', action='store_true')
    ap.add_argument('--cparse', action='store_true')
    args = ap.parse_args()
    main(args)
