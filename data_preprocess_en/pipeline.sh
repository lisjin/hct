#!/usr/bin/env bash
python proc_corpus.py  # canard/<split>.tsv and canard/<split>_pos.tsv
sh phrase_voc_optimization.sh  # canard_out/label_map.txt

python proc_corpus.py --use_context  # canard/train_valid_test_wo_context.tsv
for split in 'train' 'dev' 'test'; do
  # canard/<split>/unfound_phrs.json
  python rw_unfound.py --split ${split}

  # Lemmas: canard/<split>/unfound_lems.json
  # Parse trees: canard/<split>/cpt*.txt
  python proc_unmatch.py --split ${split} --lem --cparse

  # canard_out/<split>/{sentences|tags}.txt
  python rw_unfound.py --split ${split} --write
done
