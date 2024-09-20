import gc
import os
from time import time

import pytorch_lightning as pl

# Third party libraries
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import wandb
from dataset import generate_dataloaders, generate_transforms
from loss_function import CrossEntropyLossOneHot
from lrs_scheduler import WarmRestart

# User defined libraries
from models import se_resnext50_32x4d
from utils import init_hparams, init_logger, load_data, seed_reproducer


class CoolSystem(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        seed_reproducer(self.hparams.seed)

        self.model = se_resnext50_32x4d()
        self.criterion = CrossEntropyLossOneHot()

        self.train_outputs = []
        self.val_outputs = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
        )
        self.scheduler = WarmRestart(self.optimizer, T_max=10, T_mult=1, eta_min=1e-5)
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        step_start_time = time()
        images, labels, data_load_time = batch

        scores = self(images)
        loss = self.criterion(scores, labels)
        data_load_time = torch.sum(data_load_time)

        self.train_outputs.append(
            {
                "loss": loss,
                "data_load_time": data_load_time,
                "batch_run_time": torch.Tensor(
                    [time() - step_start_time + data_load_time]
                ).to(data_load_time.device),
            }
        )

        return loss

    def on_train_epoch_end(self):
        train_loss_mean = torch.stack(
            [output["loss"] for output in self.train_outputs]
        ).mean()
        self.train_outputs.clear()

        self.log("train_loss", train_loss_mean)

    def validation_step(self, batch, batch_idx):
        step_start_time = time()
        images, labels, data_load_time = batch
        data_load_time = torch.sum(data_load_time)
        scores = self(images)
        loss = self.criterion(scores, labels)

        self.val_outputs.append(
            {
                "val_loss": loss,
                "scores": scores,
                "labels": labels,
                "data_load_time": data_load_time,
                "batch_run_time": torch.Tensor(
                    [time() - step_start_time + data_load_time]
                ).to(data_load_time.device),
            }
        )

        return loss

    def on_validation_epoch_end(self):
        val_loss_mean = torch.stack(
            [output["val_loss"] for output in self.val_outputs]
        ).mean()
        self.data_load_times = torch.stack(
            [output["data_load_time"] for output in self.val_outputs]
        ).sum()
        self.batch_run_times = torch.stack(
            [output["batch_run_time"] for output in self.val_outputs]
        ).sum()

        scores_all = torch.cat([output["scores"] for output in self.val_outputs]).cpu()
        labels_all = torch.round(
            torch.cat([output["labels"] for output in self.val_outputs]).cpu()
        )
        val_roc_auc = roc_auc_score(labels_all, scores_all)

        self.val_outputs.clear()

        self.log("val_loss", val_loss_mean)
        self.log("val_roc_auc", val_roc_auc, prog_bar=True)


if __name__ == "__main__":
    seed_reproducer(2020)

    hparams = init_hparams()

    data, test_data = load_data()

    transforms = generate_transforms(hparams.image_size)

    # Cross-validation
    valid_roc_auc_scores = []
    folds = KFold(n_splits=5, shuffle=True, random_state=hparams.seed)
    for fold_i, (train_index, val_index) in enumerate(folds.split(data)):
        logger = init_logger()
        hparams.fold_i = fold_i
        train_data = data.iloc[train_index, :].reset_index(drop=True)
        val_data = data.iloc[val_index, :].reset_index(drop=True)

        train_dataloader, val_dataloader = generate_dataloaders(
            hparams, train_data, val_data, transforms
        )

        # Define callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor="val_roc_auc",
            save_top_k=6,
            mode="max",
            dirpath=os.path.join(hparams.log_dir, f"fold{fold_i}"),
            filename="{epoch}-{val_loss:.4f}-{val_roc_auc:.4f}",
        )
        early_stop_callback = EarlyStopping(
            monitor="val_roc_auc", patience=10, mode="max", verbose=True
        )

        # Instantiation of models and trainers
        model = CoolSystem(hparams)
        trainer = pl.Trainer(
            devices=hparams.gpus,
            accelerator="gpu",
            min_epochs=20,
            max_epochs=hparams.max_epochs,
            callbacks=[early_stop_callback, checkpoint_callback],
            precision=hparams.precision,
            num_sanity_val_steps=0,
            profiler=False,
            enable_model_summary=False,
            gradient_clip_val=hparams.gradient_clip_val,
            logger=logger,
        )

        trainer.fit(model, train_dataloader, val_dataloader)

        best_score = checkpoint_callback.best_model_score
        valid_roc_auc_scores.append(round(best_score.item(), 4))

        del model
        gc.collect()
        torch.cuda.empty_cache()
        wandb.finish()
