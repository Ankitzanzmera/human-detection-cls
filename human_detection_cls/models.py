import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import DenseNet121_Weights


class DensnetModel(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super(DensnetModel, self).__init__()
        
        self.model = models.densenet121(weights = DenseNet121_Weights.DEFAULT if pretrained else None)
        
        classifier_layer_in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features = classifier_layer_in_features, out_features = num_classes, bias = True)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x
        
        