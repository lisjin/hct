#!/usr/bin/env bash
if [ "$#" -ne 1 ]; then
    echo "Please provide dataset as argument: [canard | mudoco | rewrite]"
    exit 1
fi
TASK="$1"
[ "$TASK" = "rewrite" ] && LANG="zh" || LANG="en"
DATA_DIR="data_preprocess_${LANG}/${TASK}"
MODEL_DIR="experiments/${TASK}21_08-15"
DOMAIN_SUF="_calling"  # for MuDoCo domain adaptation
python evaluate.py \
    --dataset "${DATA_DIR}_out" \
    --model  $MODEL_DIR\
    --rule_path "${DATA_DIR}/train/rule_affinity${DOMAIN_SUF}.txt" \
    --gpu 3 \
    --restore_dir "${MODEL_DIR}/21" \
    --domain_rng_path "${DATA_DIR}/domain_rng.json"
