import torchvision.transforms as transforms
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# RESIZE_WIDTH = 224
# RESIZE_HEIGHT = 224
RESIZE_WIDTH = 96 # 256 96
RESIZE_HEIGHT = 96 #256 96
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MAX_PIX_VAL = 255.0

OUTPUT_NODES = 6 # 42
KEYPOINT_INDEXES = [8,12,16]
EFF_NET_FC_HIDDEN_UNIT = 2560 # b7 -> 2560, b0 -> 1280  

SAVE_DIR = "output/b7/8"
SAVE_TEST_RESULT_DIR = "img_out/5_new_3"
TRAIN_SET_PATH = "./keypoint_dataset/train_smallest" # train | train_smallest
TEST_SET_PATH = "vid_out/5" # "./keypoint_dataset/test" | "vid_out"

PRETRAINED_LOAD_MODEL_NAME = "efficientnet-b7"
# PRETRAINED_LOAD_MODEL_NAME = "efficientnet-b0"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 3*1e-4
WEIGHT_DECAY = 5e-4
LR_SCHEDULER_PATIENCE = 10
LR_SCHEDULER_FACTOR = 0.1
BATCH_SIZE = 32 # custom_conv_net -> [1024 for h,w = 96,96] | eff_net_b0 -> [256 for h,w = 96,96] |  eff_net_b7 -> [32 for h,w = 96,96]
NUM_EPOCHS = 1000

# NUM_WORKERS = 4
# CHECKPOINT_FILE = "custom_cnn.pth"
# CHECKPOINT_FILE = "efficientnet-b0-355c32eb.pth"
CHECKPOINT_FILE = "efficientnet-b7-dcc49843.pth"

SAVE_EPOCHS = 100
PIN_MEMORY = True
SAVE_MODEL = True
LOAD_MODEL = False

# custom_transform = transforms.Compose(
#     [
#     transforms.Resize((96, 96)),
#     transforms.PILToTensor()
#     ]
# )

custom_test_transform = A.Compose(
    [
        A.Resize(width=RESIZE_WIDTH, height=RESIZE_HEIGHT),
        A.Normalize(
            mean=MEAN,
            std=STD,
            max_pixel_value=MAX_PIX_VAL,
        ),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
)

custom_transform = A.Compose(
    [
        A.Resize(width=RESIZE_WIDTH, height=RESIZE_HEIGHT),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.8),
        A.RandomBrightnessContrast(contrast_limit=0.5, brightness_limit=0.5, p=0.2),
        A.OneOf([
            A.GaussNoise(p=0.8),
            A.CLAHE(p=0.8),
            A.ImageCompression(p=0.8),
            A.RandomGamma(p=0.8),
            A.Posterize(p=0.8),
            A.Blur(p=0.8),
        ], p=1.0),
        A.OneOf([
            A.GaussNoise(p=0.8),
            A.CLAHE(p=0.8),
            A.ImageCompression(p=0.8),
            A.RandomGamma(p=0.8),
            A.Posterize(p=0.8),
            A.Blur(p=0.8),
        ], p=1.0),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.2, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(
            mean=MEAN,
            std=STD,
            max_pixel_value=MAX_PIX_VAL,
        ),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
)