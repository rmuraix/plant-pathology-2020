import json

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from scipy.special import softmax
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm

# User defined libraries
from dataset import PlantDataset, generate_transforms
from train import CoolSystem
from utils import init_hparams, init_logger, load_data, seed_reproducer

if __name__ == "__main__":
    # Make experiment reproducible
    seed_reproducer(2020)

    # Init Hyperparameters
    hparams = init_hparams()

    # init logger
    logger = init_logger()

    # Load data
    data, test_data = load_data()

    # Generate transforms
    try:
        with open("augmentation_best_params.json", "r") as f:
            augmentation_best_params = json.load(f)

        transforms = generate_transforms(
            hparams.image_size,
            augmentation_best_params["brightness_limit"],
            augmentation_best_params["contrast_limit"],
            augmentation_best_params["brightness_contrast_p"],
            augmentation_best_params["motion_blur_limit"],
            augmentation_best_params["median_blur_limit"],
            augmentation_best_params["gaussian_blur_limit"],
            augmentation_best_params["blur_p"],
            augmentation_best_params["vertical_flip_p"],
            augmentation_best_params["holizontal_flip_p"],
            augmentation_best_params["shift_limit"],
            augmentation_best_params["scale_limit"],
            augmentation_best_params["rotate_limit"],
        )
    except FileNotFoundError:
        transforms = generate_transforms(
            hparams.image_size,
        )

    early_stop_callback = EarlyStopping(
        monitor="val_roc_auc", patience=10, mode="max", verbose=True
    )

    # Instance Model, Trainer and train model
    model = CoolSystem(hparams)
    trainer = pl.Trainer(
        devices=hparams.gpus,
        accelerator="gpu",
        min_epochs=20,
        max_epochs=hparams.max_epochs,
        callbacks=[early_stop_callback],
        enable_progress_bar=False,
        precision=hparams.precision,
        num_sanity_val_steps=0,
        profiler=False,
        gradient_clip_val=hparams.gradient_clip_val,
        logger=logger,
    )

    submission = []
    PATH = [
        "logs_submit/fold=0/epoch=57-val_loss=0.0000-val_roc_auc=0.9849.ckpt",
        "logs_submit/fold=1/epoch=48-val_loss=0.0000-val_roc_auc=0.9874.ckpt",
        "logs_submit/fold=2/epoch=69-val_loss=0.0000-val_roc_auc=0.9958.ckpt",
        "logs_submit/fold=3/epoch=55-val_loss=0.0000-val_roc_auc=0.9727.ckpt",
        "logs_submit/fold=4/epoch=69-val_loss=0.0000-val_roc_auc=0.9726.ckpt",
    ]

    folds = KFold(n_splits=5, shuffle=True, random_state=hparams.seed)
    train_data_cp = []
    for fold_i, (train_index, val_index) in enumerate(folds.split(data)):
        hparams.fold_i = fold_i
        train_data = data.iloc[train_index, :].reset_index(drop=True)
        val_data = data.iloc[val_index, :].reset_index(drop=True)
        val_data_cp = val_data.copy()
        val_dataset = PlantDataset(
            val_data,
            transforms=transforms["val_transforms"],
            soft_labels_filename=hparams.soft_labels_filename,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=hparams.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        submission = []
        model.load_state_dict(torch.load(PATH[fold_i])["state_dict"])
        model.to("cuda")
        model.eval()

        for i in range(1):
            val_preds = []
            labels = []
            with torch.no_grad():
                for image, label, times in tqdm(val_dataloader):
                    val_preds.append(model(image.to("cuda")))
                    labels.append(label)

                labels = torch.cat(labels)
                val_preds = torch.cat(val_preds)
                submission.append(val_preds.cpu().numpy())

        submission_ensembled = 0
        for sub in submission:
            submission_ensembled += softmax(sub, axis=1) / len(submission)
        val_data_cp.iloc[:, 1:] = submission_ensembled
        train_data_cp.append(val_data_cp)
    soft_labels = data[["image_id"]].merge(
        pd.concat(train_data_cp), how="left", on="image_id"
    )
    soft_labels.to_csv("soft_labels.csv", index=False)

    # ==============================================================================================================
    # Generate Submission file
    # ==============================================================================================================
    test_dataset = PlantDataset(
        test_data,
        transforms=transforms["train_transforms"],
        soft_labels_filename=hparams.soft_labels_filename,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=hparams.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    submission = []
    for path in PATH:
        model.load_state_dict(torch.load(path)["state_dict"])
        model.to("cuda")
        model.eval()

        for i in range(8):
            test_preds = []
            labels = []
            with torch.no_grad():
                for image, label, times in tqdm(test_dataloader):
                    test_preds.append(model(image.to("cuda")))
                    labels.append(label)

                labels = torch.cat(labels)
                test_preds = torch.cat(test_preds)
                submission.append(test_preds.cpu().numpy())

    submission_ensembled = 0
    for sub in submission:
        submission_ensembled += softmax(sub, axis=1) / len(submission)
    test_data.iloc[:, 1:] = submission_ensembled
    test_data.to_csv("submission.csv", index=False)
