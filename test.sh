#!/usr/bin/env bash
TASK="$1"
[ "$TASK" = "rewrite" ] && LANG="zh" || LANG="en"
DATA_DIR="data_preprocess_${LANG}/${TASK}_out"
MODEL_DIR="experiments/${TASK}21_08-11"
RULE_PATH="data_preprocess_en/${TASK}/train/rule_affinity.txt"
python evaluate.py \
        --dataset $DATA_DIR \
        --model $MODEL_DIR \
        --rule_path $RULE_PATH \
        --gpu 3 \
        --restore_dir "$MODEL_DIR/29"
