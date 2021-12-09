#!/usr/bin/env bash

BERT_VOC_PATH="chinese_L-12_H-768_A-12"
if [ ! -d $BERT_VOC_PATH ]; then
    mkdir $BERT_VOC_PATH
    curl -O https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
    unzip ${BERT_VOC_PATH}.zip
    rm -f ${BERT_VOC_PATH}/bert_*
    rm -f ${BERT_VOC_PATH}.zip
fi

DATA_DIR="rewrite"
VOCAB_F="${BERT_VOC_PATH}/vocab.txt"
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
