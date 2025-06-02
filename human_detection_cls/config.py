import os
import torch
import os.path as osp
from datetime import datetime

LABEL_MAP = {
    "no_human":0,
    "human":1,
}

artifact_dir_name = f"exp-{datetime.today().date()}"
ARTIFACT_DIR_PATH = osp.join(os.getcwd(), "artifacts", artifact_dir_name)

IMAGE_DIR = osp.join(os.getcwd(), "data/human_detection_dataset")
IMAGE_SIZE = 256
NUM_EPOCHS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CRITERION = torch.nn.BCEWithLogitsLoss()

def get_model_parameters(model):
    LR = 0.0001
    OPTIMIZER = torch.optim.AdamW(model.parameters(), lr = LR)
    return OPTIMIZER
    
    