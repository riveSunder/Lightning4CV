import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.models as tv_models
import lightning as ptl


class VGG16Classifier(ptl.LightningModule):

    def __init__(self, number_classes=10):
        super().__init__()
        
        self.encoder = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1).features
        self.num_features = 512
        self.hidden_channels = 128
        self.number_classes = number_classes


        self.head = nn.Sequential(nn.Linear(self.num_features, self.hidden_channels),\
                nn.ReLU(),\
                nn.Linear(self.hidden_channels, self.hidden_channels),\
                nn.ReLU(),\
                nn.Linear(self.hidden_channels, self.number_classes))
        
    def forward(self, x):

        # vgg16 scales input image down by 32X
        assert x.shape[-1] >= 32, "minimum image dimensions 32x32"
        assert x.shape[-2] >= 32, "minimum image dimensions 32x32"

        x = self.encoder(x).max(dim=-1)[0].max(dim=-1)[0]
        x = self.head(x) 

        return x

