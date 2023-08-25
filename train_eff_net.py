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
from utils import load_checkpoint, save_checkpoint
from dataset import keypoint_dataset
from config import custom_transform, custom_test_transform
import config
from utils import save_single_image
import math


def train_one_epoch(train_loader, val_loader, model, optimizer, loss_fn, scaler, device, scheduler, epoch, num_train_batches, num_val_batches):
    for phase in ["train", "val"]:
        losses = []
        if phase == "train":
            loader = train_loader
            model.train()
        else:
            loader = val_loader
            model.eval()
        loop = tqdm(loader)
        # num_examples = 0
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            with torch.set_grad_enabled(phase == 'train'):
                with torch.cuda.amp.autocast():
                    scores = model(data)
                    scores[targets == -1] = -1
                    loss = loss_fn(scores, targets)/data.size(0)
                    # num_examples += torch.numel(scores[targets != -1])
                    losses.append(loss.item())

                if phase == 'train':
                    # backward
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            loop.set_description(f"Epoch [{epoch}/{config.NUM_EPOCHS}]")
            loop.set_postfix(loss=f"{loss.item():.4f}")
            if batch_idx in [0,1,2] and phase == "train" and epoch%100 == 0:
                save_single_image(data, targets, ind=batch_idx)
                save_single_image(data, scores.detach(), ind=batch_idx)
            if batch_idx in [0,1,2] and phase == "val" and epoch%100 == 0:
                save_single_image(data, targets, ind=batch_idx)
                save_single_image(data, scores.detach(), ind=batch_idx)
        if phase == "train":
            scheduler.step(sum(losses)/num_train_batches)
        if phase == "train":
            print(f"{phase.capitalize()} loss average over epoch: {sum(losses)/num_train_batches}")
        else:
            print(f"{phase.capitalize()} loss average over epoch: {sum(losses)/num_val_batches}")
    if phase == "val":
        return sum(losses)/num_val_batches, model, optimizer


def main():
    config.LOAD_MODEL = False
    if config.SAVE_DIR != "":
        os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    dataset = keypoint_dataset(data_dir=config.TRAIN_SET_PATH, transform=custom_transform)

    total_size = len(dataset)
    train_size = int(total_size*0.85)

    num_train_batches = math.ceil(train_size/config.BATCH_SIZE)
    num_val_batches = math.ceil((total_size-train_size)/config.BATCH_SIZE)

    train_set, val_set = random_split(dataset, [train_size, total_size-train_size])

    train_loader = DataLoader(dataset=train_set, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY, shuffle=False)

    loss_fn = nn.MSELoss(reduction="sum")
    model = EfficientNet.from_pretrained(config.PRETRAINED_LOAD_MODEL_NAME)

    # for param in model.parameters():
    #     param.requires_grad = False

    model._fc = nn.Linear(config.EFF_NET_FC_HIDDEN_UNIT, config.OUTPUT_NODES)
    model = model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=config.LR_SCHEDULER_FACTOR, patience=config.LR_SCHEDULER_PATIENCE, verbose=True
    )


    if config.LOAD_MODEL and f"best_{config.CHECKPOINT_FILE}" in os.listdir():
        load_checkpoint(torch.load(f"best_{config.CHECKPOINT_FILE}"), model, optimizer, config.LEARNING_RATE)

    min_loss = np.inf
    for epoch in range(config.NUM_EPOCHS):
        current_loss, model, optimizer = train_one_epoch(train_loader, val_loader, model, optimizer, loss_fn, scaler, config.DEVICE, scheduler, epoch+1, num_train_batches, num_val_batches)

        # get on validation
        if config.SAVE_MODEL and min_loss > current_loss:
            min_loss = current_loss
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if os.path.isfile(os.path.join(config.SAVE_DIR, f"best_{config.CHECKPOINT_FILE}")):
                os.remove(os.path.join(config.SAVE_DIR, f"best_{config.CHECKPOINT_FILE}"))
            save_checkpoint(checkpoint, filename=os.path.join(config.SAVE_DIR, f"best_{config.CHECKPOINT_FILE}"))
        
        if config.SAVE_MODEL and (epoch+1)%config.SAVE_EPOCHS == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if os.path.isfile(f"epoch_{epoch+1}_best_{config.CHECKPOINT_FILE}"):
                os.remove(f"epoch_{epoch+1}_best_{config.CHECKPOINT_FILE}")
            save_checkpoint(checkpoint, filename=os.path.join(config.SAVE_DIR, f"epoch_{epoch+1}_best_{config.CHECKPOINT_FILE}"))

if __name__ == "__main__":
    main()
