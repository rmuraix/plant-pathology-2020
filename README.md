# Plant Pathology 2020 - FGVC7

## About

My solution to the [Plant Pathology 2020 - FGVC7](https://www.kaggle.com/c/plant-pathology-2020-fgvc7) competition on Kaggle.

As it was a late submission, I tried to improve on the [first place solution](https://github.com/ant-research/cvpr2020-plant-pathology).

## Score

- Private Score: 0.96834
- Public Score: 0.96732

## Usage

### Install dependencies

```bash
poetry install
poetry run wandb login
```
You can also use DevContainer.

### Execute

**Step 1**: Train the model using k-fold cross validation(k=5).

```bash
python train.py --train_batch_size 32 --gpus 0
```

**Step 2**: Generate soft labels for self-distillation training.

```bash
python generate_soft_labels.py
```

**Step 3**: Use soft and hard labels and train the model using k-fold cross validation(k=5).

```bash
python train.py --train_batch_size 32 --gpus 0 --soft_labels_filename soft_labels.csv --log_dir logs_submit_distill
```

**Step 4**: Generate the results of the model predictions generated by the distillation.

```bash
python generate_distill_submission.py
```

**Step 5**: Generate final results

```bash
python generate_final_submission.py
```

## Contributing

Your contribution is always welcome. Please read [Contributing Guide](https://github.com/rmuraix/.github/blob/main/.github/CONTRIBUTING.md).
