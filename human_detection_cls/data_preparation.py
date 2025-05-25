import os
import os.path as osp
import pandas as pd
from sklearn.model_selection import train_test_split

def make_csv(image_dir, label_map):
    image_paths, image_labels = [], []

    for class_name in os.listdir(image_dir):
        class_path = osp.join(image_dir, class_name)
        for image_name in os.listdir(class_path):
            full_image_path = osp.join(class_path, image_name)

            if osp.exists(full_image_path):
                image_paths.append(full_image_path)
                image_labels.append(class_name)
                
    df = pd.DataFrame({"image_path":image_paths, "labels":image_labels})
    train_csv, valid_csv = train_test_split(df, test_size=0.2, stratify=df['labels'])
    return train_csv, valid_csv
    
    
