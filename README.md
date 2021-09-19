# Hierarchical Context Tagging for Utterance Rewriting

## Data Preprocessing
To extract labels and rules automatically, run the following commands. For Chinese Rewrite dataset, replace `data_preprocess_en` with `data_preprocess_zh`. You can replace `pipeline_canard.sh` with either `pipeline_mudoco.sh` or `pipeline.sh` (for Rewrite).

```bash
cd data_preprocess_en
sh pipeline_canard.sh
```

## Training
Return to the root directory and modify line 22 of `train.sh` with the correct model directory in `experiments/` that contains `params.json`. Then run `sh train.sh <dataset>`. The top-2 checkpoints will be saved in this given directory under the current epoch number (e.g., `experiments/canard/17`).

## Evaluation
Modify line 16 of `test.sh` to point to the correct model directory. Run `test.sh <dataset> <epoch>` to evaluate using the checkpoint at a certain epoch.
