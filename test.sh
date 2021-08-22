DATASET='mudoco'
MODEL_DIR="${DATASET}21_08-22_rast"
python evaluate.py \
        --dataset ${DATASET}_out \
        --model ${MODEL_DIR}\
        --gpu 1 \
        --restore_dir "experiments/${MODEL_DIR}/5"
