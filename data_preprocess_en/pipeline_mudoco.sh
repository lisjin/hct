#!/usr/bin/env bash
DATA_DIR="mudoco"

# mudoco/<split>.tsv and mudoco/<split>_pos.tsv
# mudoco/train_valid_test_wo_context.tsv
python proc_corpus_mudoco.py --with_context --wo_context

sh phrase_voc_optimization.sh $DATA_DIR  # mudoco_out/label_map.txt

for split in 'train' 'dev' 'test'; do
    # mudoco/<split>/unfound_phrs.json
    python rw_unfound.py --split ${split} --data_dir $DATA_DIR --data_out_dir "${DATA_DIR}_out"

    # Lemmas: mudoco/<split>/unfound_lems.json
    # Parse trees: mudoco/<split>/cpt*.txt
    python proc_unmatch.py --split ${split} --lem --cparse --data_dir $DATA_DIR

    # Add --domain_rng_path "${DATA_DIR}/domain_rng.json" to below two scripts
    # for 'calling' train domain only
    python rule_extract.py --split ${split} --max_sp_width 2 --min_rule_prop .005 --data_dir $DATA_DIR

    # mudoco_out/<split>/{sentences|tags}.txt
    python rw_unfound.py --split ${split} --write --data_dir $DATA_DIR --data_out_dir "${DATA_DIR}_out"
done
