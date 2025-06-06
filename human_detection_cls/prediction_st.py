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
    def __init__(self):

        self.model = DensnetModel(num_classes=1, pretrained=False)
        self.model.load_state_dict(torch.load("saved_model/best_model.pth"))
        _, self.transforms = get_transforms(IMAGE_SIZE)
        self.reverse_label_map = {v:k for k,v in LABEL_MAP.items()}
        
    def pred(self, rgb_image):
        rgb_image = np.array(rgb_image)
        if self.transforms:
            image = self.transforms(image = rgb_image.copy())['image'].unsqueeze(0)
        
        y_pred_prob = self.model(image)
        y_pred = 1 if y_pred_prob.item() >= 0.5 else 0  
        
        grad_cam_image = self.generate_grad_cam(rgb_image, image, y_pred)    
        return y_pred_prob.item(), self.reverse_label_map[y_pred], grad_cam_image
        
    def generate_grad_cam(self, rgb_image, input_image, y_pred):
        
        rgb_image = cv2.resize(rgb_image, (IMAGE_SIZE, IMAGE_SIZE))
        rgb_image = rgb_image / 255.0
        
        target_layer = [self.model.model.features.denseblock4.denselayer16.conv2]
        targets = [BinaryClassifierOutputTarget(y_pred)]
        
        with GradCAMPlusPlus(model = self.model, target_layers=target_layer) as cam:
            grascale_cam = cam(input_tensor = input_image, targets=targets)
            grascale_cam = grascale_cam[0, :]
            grad_cam_img =  show_cam_on_image(rgb_image, grascale_cam, use_rgb=True)
        
        return grad_cam_img
