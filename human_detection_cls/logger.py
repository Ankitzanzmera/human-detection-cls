import logging
import os.path as osp

def get_training_log_config(log_dir):
    logger = logging.basicConfig(
        filename=osp.join(log_dir, "training.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode = "w"
    )
    return logging.getLogger()