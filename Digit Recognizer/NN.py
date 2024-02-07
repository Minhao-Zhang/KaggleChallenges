import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# check if cuda available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set random seed
torch.manual_seed(1)

# Load data
train = pd.read_csv('data/train.csv')
y = train['label']
X = train.drop('label', axis=1)
test = pd.read_csv('data/test.csv')

# use these data to create data loader
train_loader = DataLoader

# create a cnn model


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16,
                              kernel_size=3, stride=1, padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32,
                              kernel_size=3, stride=1, padding=0)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32*5*5, 10)
        self.batchnorm3 = nn.BatchNorm1d(10)
        self.fc2 = nn.Linear(10, 10)
        self.batchnorm4 = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(10, 10)
        self.batchnorm5 = nn.BatchNorm1d(10)
        self.fc4 = nn.Linear(10, 10)
        self.batchnorm6 = nn.BatchNorm1d(10)
        self.fc5 = nn.Linear
