# Hierarchical Context Tagging for Utterance Rewriting

## Data Preprocessing
To extract labels and rules automatically, run the following commands. For Chinese Rewrite dataset, replace `data_preprocess_en` with `data_preprocess_zh`. You can replace `pipeline_canard.sh` with either `pipeline_mudoco.sh` or `pipeline.sh` (for Rewrite).

### English
```bash
cd data_preprocess_en
BERT_VOC_PATH="uncased_L-12_H-768_A-12"
if [ ! -d $BERT_VOC_PATH ]; then
    mkdir $BERT_VOC_PATH
    curl -o ${BERT_VOC_PATH}/vocab.txt https://huggingface.co/google/bert_uncased_L-12_H-768_A-12/resolve/main/vocab.txt
fi
sh pipeline_canard.sh
```

### Chinese
```bash
cd data_preprocess_zh
BERT_VOC_PATH="chinese_L-12_H-768_A-12"
if [ ! -d $BERT_VOC_PATH ]; then
    mkdir $BERT_VOC_PATH
    curl -O https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
    unzip ${BERT_VOC_PATH}.zip
    rm -f ${BERT_VOC_PATH}/bert_*
    rm -f ${BERT_VOC_PATH}.zip
fi
sh pipeline.sh
```

## Training
Return to the root directory and modify line 22 of `train.sh` with the correct model directory in `experiments/` that contains `params.json`. Then run `sh train.sh <dataset>`. The top-2 checkpoints will be saved in this given directory under the current epoch number (e.g., `experiments/canard/17`).

## Evaluation
Modify line 16 of `test.sh` to point to the correct model directory. Run `test.sh <dataset> <epoch>` to evaluate using the checkpoint at a certain epoch.
