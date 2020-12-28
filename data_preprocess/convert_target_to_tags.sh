export BERT_BASE_DIR=chinese_L-12_H-768_A-12
export DATA_DIR=data

python preprocess_main_out.py \
  --input_file=${DATA_DIR}/test.tsv \
  --input_format=wikisplit \
  --label_map_file=${OUTPUT_DIR}/label_map.txt \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --output_arbitrary_targets_for_infeasible_examples=false

