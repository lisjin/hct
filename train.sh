MODEL_DIR="experiments/canard21_07-22"
python train.py \
        --dataset canard_out \
        --model $MODEL_DIR \
        --gpu 0 \
        --bleu_rl \
        --restore_dir "${MODEL_DIR}/13"
