import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


class MLPRegressor3(nn.Module):
    def __init__(self, input_dim=20):
        super(MLPRegressor3, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # Adjust the input dimension dynamically
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.fc5(x)

class CNNRegressor3(nn.Module):
    def __init__(self, input_dim=20):
        super(CNNRegressor3, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)  # Input channels, output channels, kernel size
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * input_dim, 128)  # Adjust size based on output of last conv layer
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension here in the forward pass
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
    
class MLPRegressor4(nn.Module):
    def __init__(self, input_dim=33):
        super(MLPRegressor4, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # Adjust the input dimension dynamically
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.relu(self.fc4(x))
        return self.fc5(x)
    
    
class CNNRegressor4(nn.Module):
    def __init__(self, input_dim=33):
        super(CNNRegressor4, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)  # Input channels, output channels, kernel size
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * input_dim, 128)  # Adjust size based on output of last conv layer
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension here in the forward pass
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)