#!/usr/bin/env bash

# canard/<split>.tsv and canard/<split>_pos.tsv
# canard/train_valid_test_wo_context.tsv
python proc_corpus_canard.py --with_context --wo_context

sh phrase_voc_optimization.sh  # canard_out/label_map.txt

python proc_corpus.py --use_context  # canard/train_valid_test_wo_context.tsv
for split in 'train' 'dev' 'test'; do
    # canard/<split>/unfound_phrs.json
    python rw_unfound.py --split ${split}

    # Lemmas: canard/<split>/unfound_lems.json
    # Parse trees: canard/<split>/cpt*.txt
    python proc_unmatch.py --split ${split} --lem --cparse

    python rule_extract.py --split ${split} --max_sp_width 3 --min_rule_prop .005

    # canard_out/<split>/{sentences|tags}.txt
    python rw_unfound.py --split ${split} --write
done