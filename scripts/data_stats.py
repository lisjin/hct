#!/usr/bin/env python3
import argparse
import os

from data_preprocess_en.utils_data import yield_sources_and_targets


def main(args):
    base = f'data_preprocess_{args.lang}/{args.dname}'
    stats_dct = {k: 0 for k in ('ctx', 'tgt', 'src')}
    match = 0
    n = 0
    for split in ('train', 'dev', 'test'):
        for src, tgt in yield_sources_and_targets(os.path.join(base, f'{split}.tsv')):
            ctx, src = src[0].split(' [CI] ')
            stats_dct['ctx'] += len(ctx.split())
            stats_dct['src'] += len(src.split())
            stats_dct['tgt'] += len(tgt.split())
            match += int(src == tgt)
            n += 1
    for k in stats_dct.keys():
        print(f'{k}:\t{(stats_dct[k] / n):.1f}')
    print(f'non-rewrite %:\t{(match / n):.4f}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--lang', default='en')
    ap.add_argument('--dname', default='canard')
    args = ap.parse_args()
    main(args)
