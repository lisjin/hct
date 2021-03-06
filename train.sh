#!/usr/bin/env bash
if [ "$#" -ne 1 ]; then
    echo "Please provide dataset as argument: [canard | mudoco | rewrite]"
    exit 1
fi
TASK="$1"
[ "$TASK" = "rewrite" ] && LANG="zh" || LANG="en"
DATA_DIR="data_preprocess_${LANG}/${TASK}"

# For domain adaptation
USE_DOM=0
DOMAIN_SUF=""
DOMAIN_RNG_PATH=""
if [[ "$TASK" = "mudoco" && $USE_DOM -eq 1 ]]; then
    DOMAIN_SUF="_calling"
    DOMAIN_RNG_PATH="${DATA_DIR}/domain_rng.json"
fi

F_SUF=""
python train.py \
    --dataset "${DATA_DIR}_out" \
    --model "experiments/${TASK}21_$(date +%m-%d)${DOMAIN_SUF}${F_SUF}" \
    --rule_path "${DATA_DIR}/train/rule_affinity${DOMAIN_SUF}${F_SUF}.txt" \
    --gpu $CUDA_VISIBLE_DEVICES \
    --f_suf $F_SUF \
    --domain_rng_path $DOMAIN_RNG_PATH \
    --restore_dir "experiments/${TASK}21_$(date +%m-%d)${DOMAIN_SUF}${F_SUF}_no-rl/01" \
    --bleu_rl
