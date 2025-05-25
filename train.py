
import os
import os.path as osp
from datetime import datetime
from torch.utils.data import DataLoader

from human_detection_cls.config import *
from human_detection_cls.data_preparation import make_csv
from human_detection_cls.utils import plot_label_counts
from human_detection_cls.logger import get_training_log_config
from human_detection_cls.datasets import BinarDataset
from human_detection_cls.augmentation import get_transforms


if __name__ == "__main__":
    artifact_dir_name = f"exp-{datetime.today().date()}"
    artifact_dir = osp.join(os.getcwd(), "artifacts", artifact_dir_name)
    os.makedirs(artifact_dir, exist_ok=True)
    
    logger = get_training_log_config(artifact_dir)
    logger.info(f"Artifact_dir path : {artifact_dir}")
    
    train_csv, valid_csv = make_csv(IMAGE_DIR, LABEL_MAP)
    logger.info(f"Train Label Counts : {train_csv['labels'].value_counts().to_dict()}")
    logger.info(f"Train Label Counts : {valid_csv['labels'].value_counts().to_dict()}")
    plot_label_counts(train_csv, artifact_dir, flag = "train")
    plot_label_counts(valid_csv, artifact_dir, flag = "valid")
    
    
    train_transform, valid_transform = get_transforms(size = IMAGE_SIZE)
    train_dataset = BinarDataset(train_csv, LABEL_MAP, train_transform)
    valid_dataset = BinarDataset(valid_csv, LABEL_MAP, train_dataset)
    
    train_dataloaders = DataLoader(train_dataset, shuffle = True, drop_last = True, pin_memory = True, batch_size = 10)
    valid_dataloaders = DataLoader(valid_dataset, shuffle = True, drop_last = True, pin_memory = True, batch_size = 10)
    
    

    
    
    

    
    
    
    
    