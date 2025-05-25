import os
import os.path as osp

LABEL_MAP = {
    "no_human":0,
    "human":1,
}

IMAGE_DIR = osp.join(os.getcwd(), "data/human_detection_dataset")

IMAGE_SIZE = 256