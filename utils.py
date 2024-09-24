import os
import random
import re
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.loggers import WandbLogger

IMG_SHAPE = (1365, 2048, 3)
# IMAGE_FOLDER = "/home/public_data_center/kaggle/plant_pathology_2020/images"
IMAGE_FOLDER = "data/images"
NPY_FOLDER = "/home/public_data_center/kaggle/plant_pathology_2020/npys"
LOG_FOLDER = "logs"


def mkdir(path: str):
    """Create directory.

    Create directory if it is not exist, else do nothing.

    Parameters
    ----------
    path: str
       Path of your directory.

    Examples
    --------
    mkdir("data/raw/train/")
    """
    try:
        if path is None:
            pass
        else:
            os.stat(path)
    except Exception:
        os.makedirs(path)


def seed_reproducer(seed=2020):
    """Reproducer for pytorch experiment.

    Parameters
    ----------
    seed: int, optional (default = 2019)
        Radnom seed.

    Example
    -------
    seed_reproducer(seed=2019).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True


def init_hparams():
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "-backbone", "--backbone", type=str, default="se_resnext50_32x4d"
    )
    parser.add_argument("-tbs", "--train_batch_size", type=int, default=32 * 1)
    parser.add_argument("-vbs", "--val_batch_size", type=int, default=16 * 1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", nargs="+", default=[480, 768])
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--max_epochs", type=int, default=70)
    parser.add_argument("--gpus", nargs="+", default=[0])
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--gradient_clip_val", type=float, default=1)
    parser.add_argument("--soft_labels_filename", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="logs_submit")
    try:
        hparams = parser.parse_args()
    except Exception:
        hparams = parser.parse_args([])
    print(type(hparams.gpus), hparams.gpus)
    if len(hparams.gpus) == 1:
        hparams.gpus = [int(hparams.gpus[0])]
    else:
        hparams.gpus = [int(gpu) for gpu in hparams.gpus]

    hparams.image_size = [int(size) for size in hparams.image_size]
    return hparams


def load_data(frac=1):
    data, test_data = (
        pd.read_csv("data/train.csv"),
        pd.read_csv("data/sample_submission.csv"),
    )
    # Do fast experiment
    if frac < 1:
        data = data.sample(frac=frac).reset_index(drop=True)
        test_data = test_data.sample(frac=frac).reset_index(drop=True)
    return data, test_data


def init_logger(project="kaggle-plant-pathology-2020"):
    logger = WandbLogger(project=project)
    return logger


def read_image(image_path):
    """Reads image data and converts to RGB format
    32.2 ms ± 2.34 ms -> self
    48.7 ms ± 2.24 ms -> plt.imread(image_path)
    """
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)


def select_best_ckpt(logs_dir):
    pattern = r"fold(\d+)/epoch=\d+-val_loss=\d+\.\d+-val_roc_auc=(\d+\.\d+)\.ckpt"

    best_checkpoints = {}

    # Loop through the files in the directory and select the best checkpoint for each fold
    for root, _, files in os.walk(logs_dir):
        for file in files:
            match = re.search(pattern, os.path.join(root, file))
            if match:
                fold = int(match.group(1))
                roc_auc = float(match.group(2))

                # Save the file with the highest roc_auc for the current fold
                if (
                    fold not in best_checkpoints
                    or roc_auc > best_checkpoints[fold]["roc_auc"]
                ):
                    best_checkpoints[fold] = {
                        "file": os.path.join(root, file),
                        "roc_auc": roc_auc,
                    }

    # Store the path of the checkpoint with the highest roc_auc in each fold in the list
    return [data["file"] for data in best_checkpoints.values()]
