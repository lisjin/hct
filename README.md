# Hierarchical Context Tagging for Utterance Rewriting

## Data Preprocessing

To extract labels and rules automatically, run the following commands. For Chinese Rewrite dataset, replace `data_preprocess_en` with `data_preprocess_zh`. You can replace `pipeline_canard.sh` with either `pipeline_mudoco.sh` or `pipeline.sh` (for Rewrite).

### Download preprocessed data

From [Google Drive](https://drive.google.com/file/d/1rsS-7SHjASAI7-hQyVkHhKdnAeWYw7lA/view?usp=sharing).
```bash
tar -xzvf RaST_data.tar.gz
mv RaST_data/canard* data_preprocess_en
mv RaST_data/mudoco* data_preprocess_en
mv RaST_data/rewrite* data_preprocess_zh
```

## Training

Return to the root directory and modify line 22 of `train.sh` with the correct model directory in `experiments/` that contains `params.json`. Then run `sh train.sh <dataset>`. The top-2 checkpoints will be saved in this given directory under the current epoch number (e.g., `experiments/canard/17`).

### Model checkpoints

Download from these links: [CANARD](https://drive.google.com/file/d/1NPbT4Wr2hu0-wVRMJO46dFxnJ2evtuau/view?usp=sharing), [MuDoCo](https://drive.google.com/file/d/1rZFJsCAJ5LiFdTVUIZEL8Wf3UITt7W2i/view?usp=sharing), [Rewrite](https://drive.google.com/file/d/1k0f9uFYE2Ncn6VEnPp74Yeug8VXqE-10/view?usp=sharing).
```bash
tar -xzvf <checkpoint_tar>  # e.g., canard21_08-08.tar.gz
mv <checkpoint_dir> experiments  # e.g., canard21_08-08
```

## Evaluation

Modify line 16 of `test.sh` to point to the correct model directory. Then run the following command:
```bash
sh test.sh <dataset> <epoch_number>  # e.g., sh test.sh canard 11
```

### From checkpoints

The best-performing models per dataset are below.

| Dataset | Path |
| --- | --- |
| CANARD | `experiments/canard21_08-08/11` |
| MuDoCo | `experiments/mudoco21_08-30/17` |
| Rewrite | `experiments/rewrite21_08_08/12` |

To evaluate on an existing checkpoint, modify line 16 of `test.sh` to point to the correct checkpoint directory. Note that the best epoch number per checkpoint can be found in the table above.
