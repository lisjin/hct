#!/usr/bin/env bash
DATA_DIR="rewrite"
VOCAB_F="chinese_L-12_H-768_A-12/vocab.txt"
python proc_corpus.py  # rewrite/<split>.tsv and rewrite/<split>_pos.tsv
sh phrase_voc_optimization.sh  # rewrite_out/label_map.txt

python proc_corpus.py --use_context  # rewrite/train_valid_test_wo_context.tsv
for split in 'train' 'dev' 'test'; do
    # rewrite/<split>/unfound_phrs.json
    python rw_unfound.py --split ${split} --vocab_f $VOCAB_F --data_dir $DATA_DIR --data_out_dir ${DATA_DIR}_out

    # Parse trees: rewrite/<split>/cpt*.txt
    python proc_unmatch.py --split ${split} --cparse --data_dir $DATA_DIR

    python rule_extract.py --split ${split} --data_dir $DATA_DIR --max_sp_width 1 --min_rule_prop .01

    # rewrite_out/<split>/{sentences|tags}.txt
    python rw_unfound.py --split ${split} --vocab_f $VOCAB_F --write --data_dir $DATA_DIR --data_out_dir ${DATA_DIR}_out --write_partial_match 0
done
