# Make results fully reproducible:
import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import random
import torch
import numpy as np

seed = 42

random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

# Import a few things we will need
import torch.nn.functional as F
import torch
from torch.optim import Adam, RAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import multiprocessing

from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

"""
Dataset
Let's start by loading our training dataset. We are going to use the Stanford Cars dataset. It consists of 196
 classes of cars with a total of 16,185 images. For this exercise we do not need any label, and we also do not 
 need a test dataset, so we are going to load both the training and the test dataset and concatenate them. 
 We are also going to transform the images to 64x64 so the exercise can complete more quickly:
"""

IMG_SIZE = 64
BATCH_SIZE = 100


def get_dataset(path):
    data_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            # We flip horizontally with probability 50%
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # Scales data into [-1,1]
            transforms.Normalize(0.5, 0.5)
        ]
    )

    train = torchvision.datasets.StanfordCars(root=path, download=True,
                                              transform=data_transform)

    test = torchvision.datasets.StanfordCars(root=path, download=True,
                                             transform=data_transform, split='test')

    return torch.utils.data.ConcatDataset([train, test])


data = get_dataset("/data/stanford_cars")
dataloader = DataLoader(
    data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
    pin_memory=True,
    num_workers=multiprocessing.cpu_count(),
    persistent_workers=True
)