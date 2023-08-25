from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import config

fc1_inp_nodes = 128*(int(config.RESIZE_WIDTH/(2**4))**2)

class keypoint_cnn(nn.Module):
    def __init__(self, output_nodes=config.OUTPUT_NODES):
        super(keypoint_cnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(fc1_inp_nodes , 1024)
        self.fc2 = nn.Linear(1024 , output_nodes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(self.dropout(x)))
        x = self.fc2(self.dropout(x))

        return x
