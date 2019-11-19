import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

USE_SIGMOID = True

#TODO add Downconv module te clean up code


class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv3_bn = nn.BatchNorm2d(128)

        self.layer1 = self.conv1
        self.layer2 = nn.Sequential()
        self.layer2.add_module("Conv2", self.conv2)
        self.layer2.add_module("BN2", self.conv2_bn)

        self.layer3 = nn.Sequential()
        self.layer3.add_module("Conv3", self.conv3)
        self.layer3.add_module("BN3", self.conv3_bn)

        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)

        self.dropout = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc1_bn = nn.BatchNorm2d(512)
        self.fc2 = nn.Linear(512, 101)

    def convs(self, x):

        if USE_SIGMOID:
            x = F.max_pool2d(F.sigmoid(self.layer1(x)), (2, 2))
            x = F.max_pool2d(F.sigmoid(self.layer2(x)), (2, 2))
            x = F.max_pool2d(F.sigmoid(self.layer3(x)), (2, 2))
        else:
            x = F.max_pool2d(F.relu(self.layer1(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.layer2(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.layer3(x)), (2, 2))


        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        #x = F.sigmoid(self.fc1_bn(self.fc1(x)))
        x = F.sigmoid(self.fc1(x))

        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
