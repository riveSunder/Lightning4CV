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

from l4cv.models.vgg16_classifier import VGG16Classifier

class ClassifierPlus(VGG16Classifier):

    def __init__(self, number_classes=2, lr=3e-4):

        super().__init__(number_classes=number_classes, lr=lr)

        self.ch = 8
        self.alpha = 1e-2
        self.ae_loss = None
        self.ae_std = None

        # dad: deep anomaly detaction
        self.dad = nn.Sequential(\
            nn.Conv2d(3, self.ch, 3, 2, padding=1),\
            nn.Conv2d(self.ch, self.ch, 3, 2, padding=1),\
            nn.Conv2d(self.ch, self.ch, 3, 2, padding=1),\
            nn.ConvTranspose2d(self.ch, self.ch, 2, 2, padding=0),\
            nn.ConvTranspose2d(self.ch, self.ch, 2, 2, padding=0),\
            nn.ConvTranspose2d(self.ch, 3, 2, 2, padding=0))

    def train_step(self, batch, batch_idx):

        data_x, labels = batch[0], batch[1]

        classification_loss = super().train_step(batch, batch_idx)

        out_x = self.dad(data_x)

        ae_loss = F.mse_loss(data_x, out_x)

        self.log("ae_loss", autoencoder_loss)

        # calculate loss statistics 
        # (exponential averaging)
        if self.ae_loss is None:
            self.ae_loss = ae_loss 
            self.ae_std = 0.0 
        else:
            alpha = self.alpha

            ae_std = torch.sqrt((self.ae_loss - ae_loss)**2)

            self.ae_std = (1-alpha) * self.ae_std + alpha * ae_std
            self.ae_loss = (1-alpha) * self.ae_loss + alpha *ae_loss 

        return loss

    def classify_and_predict(self, data_x):

        predictions = self.forward(data_x) 
        predictions = torch.softmax(predictions, dim=-1)

        recon = self.dad(data_x)

        ae_loss = torch.mean((recon - data_x)**2, dim=(1,2,3))[:,None]

        # anomalies

        if self.ae_loss is None:
            ae_mean = 0.0
            ae_std = 0.1
        else:
            ae_mean = self.ae_loss
            ae_std = self.ae_std

        anom = torch.abs(ae_loss - ae_mean)

        anom[anom > ae_std] = -1.0 
        anom[anom <= ae_std] = 1.0 

        return predictions, anom

        
def train(**kwargs):
    
    batch_size = kwargs["batch_size"]
    num_workers = kwargs["workers"]
    max_epochs = kwargs["max_epochs"]

    device = "cpu" if kwargs["device"] == "cpu" else "cuda"

    input_folder = kwargs["input_folder"]
    validation_folder =  kwargs["validation_folder"]
    test_folder =  kwargs["test_folder"]

    train_dataset = torchvision.datasets.ImageFolder(input_folder)
    validation_dataset = torchvision.datasets.ImageFolder(validation_folder)
    test_dataset = torchvision.datasets.ImageFolder(test_folder)

    train_dataset.transform = torchvision.transforms.Compose([\
            torchvision.transforms.ToTensor(),\
            torchvision.transforms.Resize((128, 128))])
    validation_dataset.transform = torchvision.transforms.Compose([\
            torchvision.transforms.ToTensor(),\
            torchvision.transforms.Resize((128, 128))])
    test_dataset.transform = torchvision.transforms.Compose([\
            torchvision.transforms.ToTensor(),\
            torchvision.transforms.Resize((128, 128))])

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
    model = ClassifierPlus(number_classes=number_classes)
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
    parser.add_argument("-v", "--validation_folder", type=str,\
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
        
   
