import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(device)

# data preparation 
train = pd.read_csv('data/new_train.csv', index_col='id')
test = pd.read_csv('data/test.csv', index_col='id')

y_columns = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
y = train[y_columns]
X = train.drop(y_columns, axis=1)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(test)
y_test = np.zeros((X_test.shape[0], len(y_columns)))
# y_test = pd.DataFrame(y_test, columns=y_columns)


X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y.values, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)


X = X.view(-1, 1, 27)  # Reshape X for training data
X_test = X_test.view(-1, 1, 27)  # Reshape X_test for testing data
# X = X.view(-1, 27)  # Reshape X for training data
# X_test = X_test.view(-1, 27)  # Reshape X_test for testing data

# Create TensorDatasets and DataLoaders for training and testing
train_dataset = TensorDataset(X, y)
test_dataset = TensorDataset(X_test, y_test)  # Assuming you have or will create y_test similarly

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# No need to shuffle the test loader
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)




# class DeeperCNN(nn.Module):
#     def __init__(self, input_channels=1, num_classes=7):
#         super(DeeperCNN, self).__init__()
#         # Convolutional layers with increased dropout
#         self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.dropout1 = nn.Dropout(0.5)  # Increased dropout
#         self.conv2 = nn.Conv1d(64, 256, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.dropout2 = nn.Dropout(0.5)  # Increased dropout
#         self.conv3 = nn.Conv1d(256, 1024, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.dropout3 = nn.Dropout(0.5)  # Increased dropout
#         self.conv4 = nn.Conv1d(1024, 512, kernel_size=3, stride=1, padding=1)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.dropout4 = nn.Dropout(0.5)  # Apply dropout also here

#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
#         self._to_linear = None
#         self._dummy_x = torch.zeros(1, input_channels, 27)
#         self._forward_features(self._dummy_x)

#         # Fully connected layers with dropout
#         self.fc1 = nn.Linear(self._to_linear, 1024)
#         self.dropout_fc1 = nn.Dropout(0.5)  # Apply dropout before final layer
#         self.fc2 = nn.Linear(1024, num_classes)

#     def _forward_features(self, x):
#         x = self.dropout1(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(x)
#         x = self.dropout2(F.relu(self.bn2(self.conv2(x))))
#         x = self.pool(x)
#         x = self.dropout3(F.relu(self.bn3(self.conv3(x))))
#         x = self.pool(x)
#         x = self.dropout4(F.relu(self.bn4(self.conv4(x))))
#         x = self.pool(x)
#         if self._to_linear is None:
#             self._to_linear = int(x.numel() / x.size(0))
#         return x

#     def forward(self, x):
#         x = self._forward_features(x)
#         x = x.view(-1, self._to_linear)
#         x = self.dropout_fc1(F.relu(self.fc1(x)))
#         x = self.fc2(x)
#         return torch.sigmoid(x)

# class ResidualBlock1D(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(ResidualBlock1D, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm1d(out_channels)
#         self.downsample = downsample
#         self.dropout = nn.Dropout(0.5)  # Add dropout

#     def forward(self, x):
#         identity = x
#         if self.downsample is not None:
#             identity = self.downsample(x)
        
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.dropout(out)  # Apply dropout
#         out += identity
#         out = self.relu(out)
#         return out



# class ResNet1D(nn.Module):
#     def __init__(self, input_channels=1, num_blocks=[2, 2, 2], num_classes=7):
#         super(ResNet1D, self).__init__()
#         self.in_channels = 64
#         self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
#         self.layer1 = self._make_layer(64, num_blocks[0])
#         self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Linear(256, num_classes)
    
#     def _make_layer(self, out_channels, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.in_channels != out_channels:
#             downsample = nn.Sequential(
#                 nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm1d(out_channels),
#             )
        
#         layers = []
#         layers.append(ResidualBlock1D(self.in_channels, out_channels, stride, downsample))
#         self.in_channels = out_channels  # Update in_channels to match out_channels for the next block
#         for _ in range(1, blocks):
#             layers.append(ResidualBlock1D(out_channels, out_channels))
        
#         return nn.Sequential(*layers)


#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
        
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
        
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return torch.sigmoid(x)

# class LargerFCNN(nn.Module):
#     def __init__(self, input_dim=27, output_dim=7, dropout_rate=0.5):
#         super(LargerFCNN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 256)  # First layer
#         self.dropout1 = nn.Dropout(dropout_rate)
#         self.fc2 = nn.Linear(256, 1024)  # Second layer
#         self.dropout2 = nn.Dropout(dropout_rate)
#         self.fc3 = nn.Linear(1024, 128)  # Third layer
#         self.dropout3 = nn.Dropout(dropout_rate)
#         self.fc4 = nn.Linear(128, 512)  # Fourth layer
#         self.dropout4 = nn.Dropout(dropout_rate)
#         self.fc5 = nn.Linear(512, output_dim)  # Output layer

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.dropout1(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout2(x)
#         x = F.relu(self.fc3(x))
#         x = self.dropout3(x)
#         x = F.relu(self.fc4(x))
#         x = self.dropout4(x)
#         x = torch.sigmoid(self.fc5(x))  # Sigmoid activation for binary output
#         return x

# class DeepCNN(nn.Module):
#     def __init__(self, num_classes=7):
#         super(DeepCNN, self).__init__()
        
#         # Convolutional layer block 1
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm1d(16)
#         self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
#         # Convolutional layer block 2
#         self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm1d(32)
#         self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
#         # Convolutional layer block 3
#         self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm1d(64)
#         self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
#         # Fully connected layers - adjust the input size according to your final tensor size after conv/pool layers
#         self.fc1 = nn.Linear(64 * 3, 120)  # Adjust '64 * 3' based on your actual output size
#         self.fc2 = nn.Linear(120, num_classes)
        
#         # Dropout layer
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         x = self.pool1(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool2(F.relu(self.bn2(self.conv2(x))))
#         x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
#         # Flatten the tensor for the fully connected layers
#         x = torch.flatten(x, 1)
        
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)  # Output raw logits for nn.CrossEntropyLoss
#         return x

class ComplexDeepCNN1D(nn.Module):
    def __init__(self, num_classes=7):
        super(ComplexDeepCNN1D, self).__init__()
        
        # Convolutional layer block 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Convolutional layer block 2
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Convolutional layer block 3
        self.conv5 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(128)
        self.conv6 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Convolutional layer block 4
        self.conv7 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm1d(256)
        self.conv8 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Fully connected layers - adjust the input size according to your final tensor size after conv/pool layers
        self.fc1 = nn.Linear(256 * 1 * 1, 512)  # Adjust '256 * 1 * 1' based on your actual output size
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool4(x)
        
        # Flatten the tensor for the fully connected layers
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)  # Output raw logits for nn.CrossEntropyLoss
        return x


# Ensure model is compatible with CUDA
model = ComplexDeepCNN1D().to(device)

# Random seed for reproducibility
torch.manual_seed(42)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)


epochs = 200
for epoch in range(epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs, labels
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')


# Store for predictions and actual labels
all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs = F.softmax(outputs, dim=1)
        all_predictions.extend(outputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

# parse all_predictions to 0 or 1
all_indexes = np.argmax(all_predictions, axis=1)
all_predictions = np.zeros(all_predictions.shape)
all_predictions[np.arange(all_indexes.size), all_indexes] = 1

# Compute AUC-ROC for each dimension
auc_scores = []
for i in range(7):  # Assuming 7 output dimensions
    auc_score = roc_auc_score(all_labels[:, i], all_predictions[:, i])
    auc_scores.append(auc_score)

mean_auc_score = np.mean(auc_scores)

print("AUC-ROC Scores for each output dimension:", auc_scores)
print("Mean AUC-ROC Score:", mean_auc_score)


# Store for predictions and actual labels
all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs = F.softmax(outputs, dim=1)
        all_predictions.extend(outputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

# parse all_predictions to 0 or 1
all_indexes = np.argmax(all_predictions, axis=1)
all_predictions = np.zeros(all_predictions.shape)
all_predictions[np.arange(all_indexes.size), all_indexes] = 1

# save to a file for submission
# id starts at 19219
submission = pd.DataFrame(all_predictions, columns=y_columns)
submission.index += 19219
submission.index.name = 'id'
submission.to_csv('submissions/cnn11.csv', index=True)