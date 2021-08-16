#!/usr/bin/env bash
if [ "$#" -ne 1 ]; then
    echo "Please provide dataset as argument: [canard | mudoco | rewrite]"
    exit 1
fi
TASK="$1"
[ "$TASK" = "rewrite" ] && LANG="zh" || LANG="en"
DATA_DIR="data_preprocess_${LANG}/${TASK}"
DOMAIN_SUF="_calling"  # for MuDoCo domain adaptation
python train.py \
    --dataset "${DATA_DIR}_out" \
    --model "experiments/${TASK}21_08-15" \
    --rule_path "${DATA_DIR}/train/rule_affinity"${DOMAIN_SUF}.txt" \
    --gpu 1 \
    --bleu_rl \
    --domain_rng_path "${DATA_DIR}/domain_rng.json"
