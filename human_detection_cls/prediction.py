import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from .config import *
from .models import DensnetModel
from .augmentation import get_transforms

class PredictionPipeline:
    def __init__(self, args):
        
        self.args = args
        self.model = DensnetModel(num_classes=1, pretrained=False)
        self.model.load_state_dict(torch.load("saved_model/best_model.pth"))
        _, self.transforms = get_transforms(IMAGE_SIZE)
        self.reverse_label_map = {v:k for k,v in LABEL_MAP.items()}
        
    def pred(self):
        rgb_image = cv2.imread(self.args.img_path)

        if self.transforms:
            image = self.transforms(image = rgb_image.copy())['image'].unsqueeze(0)
        
        y_pred = self.model(image)
        y_pred = 1 if y_pred.item() >= 0.5 else 0  
        
        if self.args.save:
            self.generate_grad_cam(rgb_image, image, y_pred)    

        return self.reverse_label_map[y_pred]
        
    def generate_grad_cam(self, rgb_image, input_image, y_pred):
        
        rgb_image = cv2.resize(rgb_image, (IMAGE_SIZE, IMAGE_SIZE))
        rgb_image = rgb_image / 255.0
        
        target_layer = [self.model.model.features.denseblock4.denselayer16.conv2]
        targets = [BinaryClassifierOutputTarget(y_pred)]
        
        with GradCAMPlusPlus(model = self.model, target_layers=target_layer) as cam:
            grascale_cam = cam(input_tensor = input_image, targets=targets)
            grascale_cam = grascale_cam[0, :]
            grad_cam =  show_cam_on_image(rgb_image, grascale_cam, use_rgb=False)
            
        cv2.imwrite(osp.join(self.args.save, osp.basename(self.args.img_path)), grad_cam)

