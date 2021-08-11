#!/usr/bin/env bash
DATA_DIR="task"
python proc_corpus_task.py  # task/<split>.tsv and task/<split>_pos.tsv
sh phrase_voc_optimization.sh $DATA_DIR  # task_out/label_map.txt

python proc_corpus_task.py --use_context  # task/train_valid_test_wo_context.tsv
for split in 'train' 'dev' 'test'; do
    # task/<split>/unfound_phrs.json
    python rw_unfound.py --split ${split} --data_dir $DATA_DIR --data_out_dir ${DATA_DIR}_out

    # Lemmas: task/<split>/unfound_lems.json
    # Parse trees: task/<split>/cpt*.txt
    python proc_unmatch.py --split ${split} --lem --cparse --data_dir $DATA_DIR

    python rule_extract.py --split ${split} --max_sp_width 1 --min_rule_prop .03 --data_dir $DATA_DIR

    # task_out/<split>/{sentences|tags}.txt
    python rw_unfound.py --split ${split} --write --data_dir $DATA_DIR --data_out_dir ${DATA_DIR}_out --write_partial_match 0
done
