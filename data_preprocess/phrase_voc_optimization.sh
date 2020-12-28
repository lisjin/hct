mkdir -p data_out
export DATA_DIR=data/
export OUTPUT_DIR=data_out

python phrase_vocabulary_optimization.py \
  --input_file=${DATA_DIR}/train_valid_test_wo_context.tsv \
  --input_format=wikisplit \
  --vocabulary_size=15000 \
  --max_input_examples=1000000 \
  --output_file=${OUTPUT_DIR}/label_map.txt
