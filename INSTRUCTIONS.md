# Instructions

## Preprocessed data

Download from Google Drive [here](https://drive.google.com/file/d/1rsS-7SHjASAI7-hQyVkHhKdnAeWYw7lA/view?usp=sharing).
```bash
tar -xzvf RaST_data.tar.gz
mv RaST_data/canard* data_preprocess_en
mv RaST_data/mudoco* data_preprocess_en
mv RaST_data/rewrite* data_preprocess_zh
```

## Model checkpoints

Download from these links: [CANARD](https://drive.google.com/file/d/1NPbT4Wr2hu0-wVRMJO46dFxnJ2evtuau/view?usp=sharing), [MuDoCo](https://drive.google.com/file/d/1rZFJsCAJ5LiFdTVUIZEL8Wf3UITt7W2i/view?usp=sharing), [Rewrite](https://drive.google.com/file/d/1k0f9uFYE2Ncn6VEnPp74Yeug8VXqE-10/view?usp=sharing).
```bash
tar -xzvf <checkpoint_tar>  # e.g., canard21_08-08.tar.gz
mv <checkpoint_dir> experiments  # e.g., canard21_08-08
```

## Evaluating from checkpoints

The best-performing models per dataset are below.

| Dataset | Path |
| --- | --- |
| CANARD | `experiments/canard21_08-08/11` |
| MuDoCo | `experiments/mudoco21_08-30/17` |
| Rewrite | `experiments/rewrite21_08_08/12` |

To evaluate on an existing checkpoint, modify line 16 of `test.sh` to point to the correct checkpoint directory. Then run the following command:
```bash
sh test.sh <dataset> <epoch_number>  # e.g., sh test.sh canard 11
```

Note that the epoch number can be found in the paths in the table above.
