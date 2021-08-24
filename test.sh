#!/usr/bin/env bash
if [ "$#" -ne 2 ]; then
    echo "Please provide dataset and epoch # as argument: [canard | mudoco | rewrite] [epoch]"
    exit 1
fi
TASK="$1"
EPOCH="$2"
[ "$TASK" = "rewrite" ] && LANG="zh" || LANG="en"
DATA_DIR="data_preprocess_${LANG}/${TASK}"

# For domain adaptation
USE_DOM=0
[ "$TASK" = "mudoco" -a $USE_DOM -eq 1 ] && DOMAIN_SUF="_calling" || DOMAIN_SUF=""
echo $DOMAIN_SUF

F_SUF="_0.015_1"
MODEL_DIR="experiments/${TASK}21_08-24${DOMAIN_SUF}${F_SUF}"
python evaluate.py \
    --dataset "${DATA_DIR}_out" \
    --model $MODEL_DIR\
    --rule_path "${DATA_DIR}/train/rule_affinity${DOMAIN_SUF}${F_SUF}.txt" \
    --gpu 2 \
    --restore_dir "${MODEL_DIR}/${EPOCH}" \
    --f_suf ${F_SUF}

if [ ! -z "$DOMAIN_SUF" ]; then
    python dom_score.py --dataset $TASK --lang $LANG --domain_suf $DOMAIN_SUF --f_suf $F_SUF --epoch $EPOCH
fi
