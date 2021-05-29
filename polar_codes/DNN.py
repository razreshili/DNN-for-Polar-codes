import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(16,64)
        self.bn1 = nn.BatchNorm1d(64, momentum=.9999)
        self.fc2 = nn.Linear(64,64)
        self.bn2 = nn.BatchNorm1d(64, momentum=.9999)
        self.fc3 = nn.Linear(64,64)
        self.bn3 = nn.BatchNorm1d(64, momentum=.9999)
        self.fc4 = nn.Linear(64,8)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = F.relu(self.fc4(x))
        return x

def Tensor_Change(ten, size, len):
    for i in range(size):
        for j in range(len):
            if ten[i][j] < 0.5:
                ten[i][j] = 0
            else:
                ten[i][j] = 1
    return ten

def number_of_errors(predicted, original):
    errors = 0
    len_of_pred = len(predicted)
    for i in range(len_of_pred):
        if predicted[i] != original[i]:
            errors += 1
    return errors
