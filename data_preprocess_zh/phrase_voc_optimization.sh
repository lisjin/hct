export DATA_DIR=rewrite
export OUTPUT_DIR=${DATA_DIR}_out
mkdir -p $OUTPUT_DIR

python3 phrase_vocabulary_optimization.py \
  --input_file=${DATA_DIR}/train_valid_test_wo_context.tsv \
  --vocabulary_size=15000 \
  --max_input_examples=1000000 \
  --output_file=${OUTPUT_DIR}/label_map.txt
