# ISSUE WITH PYTORCH INSTALL

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda

class CustomImageDataset(Dataset):
    def __init__(self, labels_file, dataset_file, transform=None, target_transform=None):
        self.image_labels = pd.read_csv(labels_file)
        self.dataset = pd.read_csv(dataset_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        image = self.dataset.iloc[idx,:]
        label = self.image_labels.iloc[idx, 0]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

project_dir = r"C:\Users\jackw\Documents\MAAE4904\Project" # Create folder for datasets, model files, results
batch_dir = os.path.join(project_dir,r"cifar-10-python\cifar-10-batches-py") # Change path to where you have downloaded batch files
X_train_path = os.path.join(project_dir,'merged_X_train_batches.csv')
y_train_path = os.path.join(project_dir,'merged_y_train_batches.csv')
X_test_path = os.path.join(project_dir,'X_test_batch.csv')
y_test_path = os.path.join(project_dir,'y_test_batch.csv')

target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
training_data = CustomImageDataset(y_train_path,X_train_path,transform=ToTensor,target_transform=target_transform)
test_data = CustomImageDataset(y_test_path,X_test_path,transform=ToTensor,target_transform=target_transform)

train_dataloader = DataLoader(training_data, batch_size=200, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=200, shuffle=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
