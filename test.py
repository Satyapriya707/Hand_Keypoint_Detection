import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from utils import load_checkpoint, save_checkpoint, save_image
from dataset import keypoint_dataset
from config import custom_transform, custom_test_transform
import config
from network import keypoint_cnn

config.LOAD_MODEL = True

test_set = keypoint_dataset(data_dir=config.TEST_SET_PATH, transform=custom_test_transform, train=False)

test_loader = DataLoader(dataset=test_set, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY, shuffle=False)

model = keypoint_cnn(output_nodes=config.OUTPUT_NODES)
model = model.to(config.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

if config.LOAD_MODEL and f"best_{config.CHECKPOINT_FILE}" in os.listdir():
    load_checkpoint(torch.load(f"best_{config.CHECKPOINT_FILE}"), model, optimizer, config.LEARNING_RATE)

model.eval()

loop = tqdm(test_loader)
for batch_idx, (data, targets) in enumerate(loop):
    data = data.to(device=config.DEVICE)
    targets = targets.to(device=config.DEVICE)

    # forward
    with torch.no_grad():
        scores = model(data)
        scores = scores.squeeze(0)
        save_image(data, scores, config.SAVE_TEST_RESULT_DIR, start_save_idx=batch_idx*scores.size(0))
