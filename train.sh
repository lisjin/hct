#!/usr/bin/env bash
if [ "$#" -ne 1 ]; then
    echo "Please provide dataset as argument: [canard | mudoco | rewrite]"
    exit 1
fi
TASK="$1"
[ "$TASK" = "rewrite" ] && LANG="zh" || LANG="en"
DATA_DIR="data_preprocess_${LANG}/${TASK}_out"
MODEL_DIR="experiments/${TASK}21_08-12"
RULE_PATH="data_preprocess_en/${TASK}/train/rule_affinity.txt"
python train.py \
    --dataset $DATA_DIR \
    --model $MODEL_DIR \
    --rule_path $RULE_PATH \
    --gpu 1 \
    --bleu_rl
