import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from PIL import Image
import config

class keypoint_dataset(Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        super().__init__()
        self.data_dir = data_dir
        self.if_train = train
        self.total_list = os.listdir(data_dir)
        if self.if_train:
            self.num = int(len(self.total_list)/2)
        else:
            self.num = len(self.total_list)
        self.transform = transform
    def __len__(self):
        return self.num
    def __getitem__(self, index):
        if self.if_train:
            img = Image.open(os.path.join(self.data_dir, self.total_list[int(2*index)])) 
            img = np.array(img, dtype=np.uint8)
            with open(os.path.join( self.data_dir, self.total_list[int(2*index + 1)]), "r") as f:
                labels = json.load(f)
            label_list = []
            for indx in config.KEYPOINT_INDEXES:
                if labels["hand_pts"][indx][2] != 0:
                    label_list.append(labels["hand_pts"][indx][0])
                    label_list.append(labels["hand_pts"][indx][1])
            label_list = torch.tensor(label_list, dtype=torch.float)
            label_list = label_list.view(-1,2)
        else:
            img = Image.open(os.path.join(self.data_dir, self.total_list[int(index)])) 
            img = np.array(img, dtype=np.uint8)
            label_list = torch.ones(2*len(config.KEYPOINT_INDEXES)).view(-1,2)
        if self.transform:
            augmentations = self.transform(image=img, keypoints=label_list)
            img = augmentations["image"]
            label_list = augmentations["keypoints"]
            label_list = torch.tensor(label_list)
        label_list = label_list.view(-1)
        return img, label_list
