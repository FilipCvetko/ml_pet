import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class PETDataset(Dataset):
    def __init__(self, data_path, label_path, x_size, y_size):
        self.x_size = x_size
        self.y_size = y_size
        self.data_path = data_path

        # read in the labels from the xlsx file
        self.labels = pd.read_excel(label_path, index_col=0)

        # create a list of all the file names
        self.file_names = []
        for folder in os.listdir(data_path):
            for file_name in os.listdir(os.path.join(data_path, folder)):
                if file_name not in self.labels.index:
                    continue
                self.file_names.append((folder, file_name))

        # create a dictionary to map labels to indices
        self.label_map = {"ds": 0, "ls": 1, "dz": 2, "lz": 3, "e": 4}

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        folder, file_name = self.file_names[index]

        # read in the pickle file
        with open(os.path.join(self.data_path, folder, file_name), "rb") as f:
            data = pickle.load(f)

        # normalize the data
        data = (data - np.mean(data)) / np.std(data)

        z_pad_size = 97 - data.shape[0]
        z_pad = np.zeros((z_pad_size, data.shape[1], data.shape[2]))
        data = np.concatenate((z_pad, data), axis=0)

        # Calculate the zoom factors for height and width dimensions
        zoom_factors = [1, self.x_size / data.shape[1], self.y_size / data.shape[2]]

        # Use ndimage zoom to resize the image along height and width dimensions
        data = np.array([ndimage.zoom(data, zoom_factors, order=1)])

        # get the corresponding label
        label = self.labels.loc[file_name, "OP lokacija"]
        if not isinstance(label, float) or not np.isnan(label):
            label = np.array(
                [self.label_map[l] for l in label.split(",") if l != "nan"]
            )
            label = torch.tensor(label)
            label = F.one_hot(label, num_classes=len(self.label_map))
        else:
            label = torch.zeros(len(self.label_map))
            label = label.unsqueeze(0)

        print(data.shape)
        return data, label


# define the hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 1

# set the preset x and y dimensions
X_SIZE = 200
Y_SIZE = 200

# set the paths for the data and label files
data_path = "/home/filip/IT/Projects/ml_pet/data/interim/25-04-2023"
label_path = "/home/filip/IT/Projects/ml_pet/data/raw/izvidi.xlsx"

# create a dataset and split it into training and testing sets
dataset = PETDataset(data_path, label_path, x_size=X_SIZE, y_size=Y_SIZE)
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# create data loaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


class PETNet(torch.nn.Module):
    def __init__(self, num_labels):
        super(PETNet, self).__init__()
        self.conv1 = torch.nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.pool1 = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.pool2 = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool3 = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(32 * 24 * 24 * 8, 128)
        self.fc2 = torch.nn.Linear(128, num_labels)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = self.pool3(x)
        x = x.view(-1, 32 * 24 * 24 * 8)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Define the training loop
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.double())
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    train_acc = 100.0 * correct / total
    return train_loss, train_acc


# Define the testing loop
def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            pred = torch.sigmoid(output) > 0.5
            acc = (pred == target).sum().item() / (target.shape[0] * target.shape[1])
            test_acc += acc
    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    print(
        "Test set: Average loss: {:.4f}, Accuracy: {:.4f}".format(test_loss, test_acc)
    )


model = PETNet(num_labels=len(dataset.label_map))
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(
    train(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )
)
print(test(model, test_loader, criterion))
