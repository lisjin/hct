MODEL_DIR="experiments/canard21_3"
python train.py \
        --dataset canard_out \
        --model $MODEL_DIR \
        --gpu 4 \
        --bleu_rl
