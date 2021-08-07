#!/usr/bin/env bash
LANG=$1
if [ "$LANG" = "en" ]; then
        data_dir="data_preprocess_en/canard_out"
        MODEL_DIR="experiments/canard21_08-04"
        RULE_PATH="data_preprocess_en/canard/train/rule_affinity.txt"
elif [ "$LANG" = "zh" ]; then
        DATA_DIR="data_preprocess_zh/rewrite_out"
        MODEL_DIR="experiments/rewrite21_08-06"
        RULE_PATH="data_preprocess_zh/rewrite/train/rule_affinity.txt"
fi
python train.py \
        --dataset $DATA_DIR \
        --model $MODEL_DIR \
        --rule_path $RULE_PATH \
        --gpu 2 \
        --bleu_rl
