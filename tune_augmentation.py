import gc
import json

import optuna
import pytorch_lightning as pl

# Third party libraries
import torch
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

import wandb
from dataset import generate_dataloaders, generate_transforms

# User defined libraries
from train import CoolSystem
from utils import init_hparams, init_logger, load_data


def objective(trial):
    hparams = init_hparams()
    brightness_limit = trial.suggest_float("brightness_limit", 0.05, 0.2)
    contrast_limit = trial.suggest_float("contrast_limit", 0.05, 0.2)
    brightness_contrast_p = trial.suggest_float("brightness_contrast_p", 0.1, 1.0)
    motion_blur_limit = trial.suggest_categorical("motion_blur_limit", [3, 5, 7])
    median_blur_limit = trial.suggest_categorical("median_blur_limit", [3, 5, 7])
    gaussian_blur_limit = trial.suggest_categorical("gaussian_blur_limit", [3, 5, 7])
    blur_p = trial.suggest_float("blur_p", 0.1, 0.5)
    vertical_flip_p = trial.suggest_float("vertical_flip_p", 0.1, 0.5)
    holizontal_flip_p = trial.suggest_float("holizontal_flip_p", 0.1, 0.5)
    shift_limit = trial.suggest_float("shift_limit", 0.1, 0.5)
    scale_limit = trial.suggest_float("scale_limit", 0.1, 0.5)
    rotate_limit = trial.suggest_float("rotate_limit", 5.0, 30.0)

    logger = init_logger(project="kaggle-plant-pathology-2020-tune")
    data, test_data = load_data()
    # train val split
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=2020)

    transforms = generate_transforms(
        hparams.image_size,
        brightness_limit,
        contrast_limit,
        brightness_contrast_p,
        motion_blur_limit,
        median_blur_limit,
        gaussian_blur_limit,
        blur_p,
        vertical_flip_p,
        holizontal_flip_p,
        shift_limit,
        scale_limit,
        rotate_limit,
    )
    train_dataloader, val_dataloader = generate_dataloaders(
        hparams, train_data, val_data, transforms
    )

    early_stop_callback = EarlyStopping(
        monitor="val_roc_auc", patience=10, mode="max", verbose=True
    )

    model = CoolSystem(hparams)
    trainer = pl.Trainer(
        devices=hparams.gpus,
        accelerator="gpu",
        min_epochs=20,
        max_epochs=hparams.max_epochs,
        callbacks=[early_stop_callback],
        precision=hparams.precision,
        num_sanity_val_steps=0,
        profiler=False,
        enable_model_summary=False,
        gradient_clip_val=hparams.gradient_clip_val,
        logger=logger,
        log_every_n_steps=45,
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    metrics = trainer.callback_metrics["val_roc_auc"].item()

    del model
    gc.collect()
    torch.cuda.empty_cache()
    wandb.finish()

    return metrics


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler()
    )
    study.optimize(objective, n_trials=50)

    print(f"Acc:{study.best_value}")

    # save hyperparameters to json
    with open("optuna_best_params.json", "w") as f:
        json.dump(study.best_params, f)
