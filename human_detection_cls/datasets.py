import cv2
import torch
import os.path as osp
from torch.utils.data import Dataset


class BinarDataset(Dataset):
    def __init__(self, df, label_map, transforms = None):
        self.df = df
        self.label_map = label_map
        self.transform = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        image_path = row['image_path']
        labels = row['labels']
        
        if osp.exists(image_path):
            rgb_image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                image = self.transform(image = rgb_image.copy())['image']
            else:
                image = rgb_image / 255.0
        
        label = torch.tensor(self.label_map[labels], dtype=torch.float)
        sample = {"image_path":image_path, "image":image, "label":label}
        
        return sample