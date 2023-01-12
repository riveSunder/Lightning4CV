import argparse
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset

import torchmetrics
import torchvision
import torchvision.models as tv_models
import lightning as pl

import matplotlib.pyplot as plt

from l4cv.models.vgg16_classifier import VGG16Classifier

class ClassifierPlus(VGG16Classifier):

    def __init__(self, number_classes=2, lr=3e-4):

        super().__init__(number_classes=number_classes, lr=lr)

        self.ch = 64
        self.alpha = 1e-2
        self.ae_loss = None
        self.ae_std = None
        self.sigma = 3
        self.dad_weight = 0.5

        # dad: deep anomaly detaction
        self.dad = nn.Sequential(\
            nn.Conv2d(3, self.ch, 3, 2, padding=1),\
            nn.ReLU(),\
            nn.Conv2d(self.ch, self.ch, 3, 2, padding=1),\
            nn.ReLU(),\
            nn.Conv2d(self.ch, self.ch, 3, 2, padding=1),\
            nn.ReLU(),\
            nn.Conv2d(self.ch, self.ch, 3, 2, padding=1),\
            nn.ReLU(),\
            nn.Conv2d(self.ch, self.ch, 3, 2, padding=1),\
            nn.ReLU(),\
            nn.Conv2d(self.ch, self.ch, 3, 2, padding=1),\
            nn.ReLU(),\
            nn.ConvTranspose2d(self.ch, self.ch, 2, 2, padding=0),\
            nn.ReLU(),\
            nn.ConvTranspose2d(self.ch, self.ch, 2, 2, padding=0),\
            nn.ReLU(),\
            nn.ConvTranspose2d(self.ch, self.ch, 2, 2, padding=0),\
            nn.ReLU(),\
            nn.ConvTranspose2d(self.ch, self.ch, 2, 2, padding=0),\
            nn.ReLU(),\
            nn.ConvTranspose2d(self.ch, self.ch, 2, 2, padding=0),\
            nn.ReLU(),\
            nn.ConvTranspose2d(self.ch, 3, 2, 2, padding=0))

    def training_step(self, batch, batch_idx):

        data_x, labels = batch[0], batch[1]

        classification_loss = super().training_step(batch, batch_idx)

        out_x = self.dad(data_x)

        ae_loss = F.mse_loss(data_x, out_x)

        self.log("ae_loss", ae_loss)

        # calculate loss statistics
        # (exponential averaging)
        if self.ae_loss is None:
            self.ae_loss = ae_loss.detach()
            self.ae_std = 0.0
        else:
            alpha = self.alpha

            ae_std = torch.sqrt((self.ae_loss - ae_loss.detach())**2)

            self.ae_std = (1-alpha) * self.ae_std + alpha * ae_std
            self.ae_loss = (1-alpha) * self.ae_loss + alpha * ae_loss.detach()

        loss = self.dad_weight * ae_loss + classification_loss
        return loss

    def classify_and_predict(self, data_x):

        self.eval()

        predictions = self.forward(data_x)
        predictions = torch.softmax(predictions, dim=-1)

        recon = self.dad(data_x)

        ae_loss = torch.mean((recon - data_x)**2, dim=(1,2,3))[:,None]

        # anomalies

        if self.ae_loss is None:
            ae_mean = 0.0
            ae_std = 0.1
        else:
            ae_mean = self.ae_loss.to(self.device)
            ae_std = self.ae_std.to(self.device)

        anom_loss = ae_loss - ae_mean
        anom = 0.0 * anom_loss

        anom[anom_loss > self.sigma * ae_std] = -1.0
        anom[anom_loss <= self.sigma * ae_std] = 1.0

        return predictions, anom

def plot_class_dad(in_tensor, out_tensor, \
        in_pred, in_anom, \
        out_pred, out_anom, \
        classes):

    fig, ax = plt.subplots(3, 3, figsize=(8,8))

    for ii in range(3):
        my_img = out_tensor[ii].permute(1,2,0).cpu().numpy()
        label = classes[torch.argmax(out_pred[ii])]
        abnormal = "'abnormal'" if out_anom[ii] < 0 else "'normal'"
        ax[ii, 2].imshow(my_img)
        ax[ii, 2].set_title(f"{abnormal} image \n predicted: {label}")

    for jj in range(3):
        for kk in range(2):

            idx = jj * 2 + kk

            my_img = in_tensor[idx].permute(1,2,0).cpu().numpy()
            label = classes[torch.argmax(in_pred[idx])]
            abnormal = "'abnormal'" if in_anom[idx] < 0 else "'normal'"
            ax[jj, kk].imshow(my_img)
            ax[jj, kk].set_title(f"{abnormal} image \n predicted: {label}")

    for ll in range(3):
        for mm in range(3):
            ax[ll,mm].set_xticklabels("")
            ax[ll,mm].set_yticklabels("")


    return fig

def train(**kwargs):

    my_seed = 13

    torch.manual_seed(13)

    batch_size = kwargs["batch_size"]
    num_workers = kwargs["workers"]
    max_epochs = kwargs["max_epochs"]
    lr = kwargs["lr"]

    device = "cpu" if kwargs["device"] == "cpu" else "cuda"

    input_folder = kwargs["input_folder"]
    anomaly_folder = kwargs["anomaly_folder"]
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
            shuffle=True,\
            num_workers=num_workers)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, \
            batch_size=batch_size, \
            shuffle=True,\
            num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, \
            batch_size=batch_size*4, \
            shuffle=True,\
            num_workers=num_workers)

    if torch.cuda.is_available() and device == "cuda":
        trainer = pl.Trainer(accelerator="gpu", \
                devices=1, max_epochs=max_epochs)
    else:
        trainer = pl.Trainer(max_epochs=max_epochs)

    number_classes = len(train_dataset.classes)

    model = ClassifierPlus(number_classes=number_classes, lr=lr)
    trainer.fit(model=model, \
            train_dataloaders=train_dataloader,\
            val_dataloaders=validation_dataloader)


    save_tag = str(int(time.time()))[-8:]
    torch.save(model.state_dict(), f"parameters/class_plus_{save_tag}.pt")

    out_dataset = torchvision.datasets.ImageFolder(anomaly_folder)

    out_dataset.transform = torchvision.transforms.Compose([\
            torchvision.transforms.ToTensor(),\
            torchvision.transforms.Resize((128, 128))])

    out_dataloader = torch.utils.data.DataLoader(\
            out_dataset, batch_size=32,\
            shuffle=True,\
            num_workers=num_workers)

    for out_batch in out_dataloader:
        break

    for in_batch in test_dataloader:
        break

    in_pred, in_anom = model.classify_and_predict(in_batch[0])
    out_pred, out_anom = model.classify_and_predict(out_batch[0])

    classes = test_dataset.classes

    fig = plot_class_dad(in_batch[0], out_batch[0],\
            in_pred, in_anom,\
            out_pred, out_anom,\
            classes)

    plt.tight_layout()
    fig.savefig(f"dad_ptl_{max_epochs}.png")

    #plt.show()

    in_accuracy = (1.0 * (torch.argmax(in_pred, dim=-1) == in_batch[1])).mean()
    out_anom_tp = (1.0 * (out_anom < 0)).sum()
    in_anom_tn = (1.0 * (in_anom > 0)).sum()

    anom_accuracy = (out_anom_tp + in_anom_tn)\
           / (in_pred.shape[0] + out_pred.shape[0])

    test_msg = f"test accuracy (cats/dogs) {in_accuracy:.4f}\n"
    test_msg += f"   true positive anomalies: {out_anom_tp}\n"
    test_msg += f"   true negative non-anomalies: {in_anom_tn}\n"
    test_msg += f"   anomaly accuracy: {anom_accuracy:.4f}\n"

    print(test_msg)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--anomaly_folder", type=str,\
            default="data/cats_and_dogs/out_set/")
    parser.add_argument("-b", "--batch_size", type=int,\
            default=32)
    parser.add_argument("-d", "--device", type=str,\
            default="cuda",\
            help="device to use: ['cpu'|'cuda','gpu']")
    parser.add_argument("-i", "--input_folder", type=str,\
            default="data/cats_and_dogs/training_set/training_set/")
    parser.add_argument("-l", "--lr", type=float,\
            default=1e-4)

    parser.add_argument("-v", "--validation_folder", type=str,\
            default="data/cats_and_dogs/training_set/val_training_set/")
    parser.add_argument("-t", "--test_folder", type=str,\
            default="data/cats_and_dogs/test_set/test_set/")
    parser.add_argument("-w", "--workers", type=int,\
            default=1)
    parser.add_argument("-m", "--max_epochs", type=int,\
            default=100)

    args = parser.parse_args()

    kwargs = dict(args._get_kwargs())

    train(**kwargs)
