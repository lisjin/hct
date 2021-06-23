export BERT_BASE_DIR=uncased_L-12_H-768_A-12
export DATA_DIR=canard
export OUTPUT_DIR=canard_out

python3 preprocess_main_out.py \
  --input_file=${DATA_DIR}/train.tsv \
  --input_format=wikisplit \
  --label_map_file=${OUTPUT_DIR}/label_map.txt \
  --tag_file=${OUTPUT_DIR}/tags.txt \
  --sen_file=${OUTPUT_DIR}/sentences.txt \
  --unfound_dct_file=${OUTPUT_DIR}/unfound_phrs.json \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --output_arbitrary_targets_for_infeasible_examples=false

