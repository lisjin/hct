DATA_DIR="data_preprocess_en/canard_out"
MODEL_DIR="experiments/canard21_08-02"
RULE_PATH="data_preprocess_en/canard/train/rule_affinity.txt"
python train.py \
        --dataset $DATA_DIR \
        --model $MODEL_DIR \
        --rule_path $RULE_PATH \
        --gpu 2 \
        --bleu_rl
