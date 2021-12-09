#!/usr/bin/env bash

BERT_VOC_PATH="uncased_L-12_H-768_A-12"
if [ ! -d $BERT_VOC_PATH ]; then
    mkdir $BERT_VOC_PATH
    curl -o ${BERT_VOC_PATH}/vocab.txt https://huggingface.co/google/bert_uncased_L-12_H-768_A-12/resolve/main/vocab.txt
fi

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
