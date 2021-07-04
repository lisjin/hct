MODEL_DIR="experiments/canard21_2"
python evaluate.py \
        --dataset canard_out \
        --model $MODEL_DIR \
        --gpu 3 \
        --restore_dir "$MODEL_DIR/8"
