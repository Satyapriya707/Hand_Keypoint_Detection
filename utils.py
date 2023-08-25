import torch
import matplotlib.pyplot as plt
import matplotlib
import config
import os

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("Saving checkpoint ", "-"*10)
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer, lr):
    print("Loading checkpoint ", "-"*10)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def save_image(images, predictions, save_dir, start_save_idx=0):
    matplotlib.use('Agg')
    os.makedirs(save_dir, exist_ok=True)
    predictions = predictions.to(device="cpu")
    for ind in range(images.size(0)):
        img_new = images[ind].permute(1,2,0)
        std_tensor = torch.tensor(config.STD).to(config.DEVICE)
        mean_tensor = torch.tensor(config.MEAN).to(config.DEVICE)
        img_new = img_new*(std_tensor*config.MAX_PIX_VAL) + (mean_tensor*config.MAX_PIX_VAL)
        img_new = img_new.int().to(device="cpu")
        npimg = img_new.numpy()
        plt.imshow(npimg)
        plt.plot(predictions[ind].numpy()[0::2], predictions[ind].numpy()[1::2], "go")
        for indx, (x, y) in enumerate(zip(predictions[ind].numpy()[0::2], predictions[ind].numpy()[1::2])):
            plt.text(x, y, str(indx), color="red", fontsize=12)
        plt.savefig(os.path.join(save_dir, f"{start_save_idx + ind}.jpg"))
        # plt.show()
        plt.close()

def save_single_image(images, predictions, ind=0):
    predictions = predictions.to(device="cpu")
    img_new = images[ind].permute(1,2,0)
    std_tensor = torch.tensor(config.STD).to(config.DEVICE)
    mean_tensor = torch.tensor(config.MEAN).to(config.DEVICE)
    img_new = img_new*(std_tensor*config.MAX_PIX_VAL) + (mean_tensor*config.MAX_PIX_VAL)
    img_new = img_new.int().to(device="cpu")
    npimg = img_new.numpy()
    plt.imshow(npimg)
    plt.plot(predictions[ind].numpy()[0::2], predictions[ind].numpy()[1::2], "go")
    for ind, (x, y) in enumerate(zip(predictions[ind].numpy()[0::2], predictions[ind].numpy()[1::2])):
        plt.text(x, y, str(ind), color="red", fontsize=12)
    plt.show()
