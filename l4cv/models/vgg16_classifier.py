import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset

import torchmetrics
import torchvision
import torchvision.models as tv_models
import lightning as pl

class VGG16Classifier(pl.LightningModule):

    def __init__(self, number_classes=10, lr=3e-4, dropout=0.25):
        super().__init__()
        
        self.encoder = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1).features
        self.num_features = 512
        self.hidden_channels = 128
        self.number_classes = number_classes
        self.lr = lr
        self.dropout = dropout


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
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.head(x) 

        return x

    def training_step(self, batch, batch_idx):

        self.zero_grad() 

        data_x, labels = batch[0], batch[1]
        predictions =  self.forward(data_x)

        loss = F.cross_entropy(predictions, labels)
        accuracy = torchmetrics.functional.accuracy(\
                predictions, labels.long())

        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)

        return loss

    def validation_step(self, batch, batch_idx):

        with torch.no_grad():

            data_x, labels = batch[0], batch[1]
            predictions =  self.forward(data_x)

            loss = F.cross_entropy(predictions, labels)
            accuracy = torchmetrics.functional.accuracy(\
                    predictions, labels.long())

            self.log("validation_loss", loss)
            self.log("validation_accuracy", accuracy)

        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), \
                lr=self.lr)

        return optimizer

def train(**kwargs):
    
    batch_size = kwargs["batch_size"]
    num_workers = kwargs["workers"]
    max_epochs = kwargs["max_epochs"]

    device = "cpu" if kwargs["device"] == "cpu" else "cuda"

    input_folder = kwargs["input_folder"]
    validation_folder =  kwargs["input_folder"]
    test_folder =  kwargs["input_folder"]

    train_dataset = torchvision.datasets.ImageFolder(input_folder)
    validation_dataset = torchvision.datasets.ImageFolder(validation_folder)
    test_dataset = torchvision.datasets.ImageFolder(test_folder)

    train_dataset.transform = torchvision.transforms.ToTensor()
    validation_dataset.transform = torchvision.transforms.ToTensor()
    test_dataset.transform = torchvision.transforms.ToTensor()

    train_dataloader = torch.utils.data.DataLoader(\
            train_dataset, batch_size=batch_size,\
            num_workers=num_workers) 
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, \
            batch_size=batch_size, \
            num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, \
            batch_size=batch_size, \
            num_workers=num_workers)
    
    if torch.cuda.is_available() and device == "cuda":
        trainer = pl.Trainer(accelerator="gpu", \
                devices=1, max_epochs=max_epochs)
    else:
        trainer = pl.Trainer(max_epochs=max_epochs)

    number_classes = len(train_dataset.classes)
    model = VGG16Classifier(number_classes=number_classes)
    trainer.fit(model=model, \
            train_dataloaders=train_dataloader,\
            val_dataloaders=validation_dataloader)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--device", type=str,\
            default="cuda",\
            help="device to use: ['cpu'|'cuda','gpu']")
    parser.add_argument("-i", "--input_folder", type=str,\
            default="data/trees")
    parser.add_argument("-v", "--validaton_folder", type=str,\
            default="data/validation_trees")
    parser.add_argument("-t", "--test_folder", type=str,\
            default="data/test_trees")
    parser.add_argument("-b", "--batch_size", type=int,\
            default=32)
    parser.add_argument("-w", "--workers", type=int,\
            default=1)
    parser.add_argument("-m", "--max_epochs", type=int,\
            default=100)

    args = parser.parse_args()

    kwargs = dict(args._get_kwargs())

    train(**kwargs)

