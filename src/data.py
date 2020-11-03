import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, sampler

import numpy as np
import os
import random
from glob import glob
from PIL import Image

batch_size = 256  # going higher than 43 results in NaN results if using SGD

normalize = False
if normalize:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])
else:
    transform = transforms.Compose([transforms.ToTensor()])

training_set_size = 50_000
val_set_size = 10_000
test_set_size = 10_000

test_set, _ = random_split(
    torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform),
    [test_set_size, 10_000 - test_set_size]
)
train_set, val_set, _ = random_split(
    torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform),
    [training_set_size, val_set_size, 60_000 - (training_set_size + val_set_size)]
)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)


notmnist_path = "../data/notMNIST_small/"
class NotMNIST:
    size = None
    misses = 0
    
    def __init__(self):
        self.files = [
            os.path.join(notmnist_path, directory, fn)
            for directory in os.listdir(notmnist_path)
            for fn in os.listdir(os.path.join(notmnist_path, directory))
        ]

        self.ims = np.zeros((len(self), 28, 28), dtype=np.float32)
        for i, fn in enumerate(self.files[:len(self)]):
            try:
                self.ims[i, :, :] = np.asarray(Image.open(fn))
            except:
                self.misses += 1

        self.labels = -np.ones(len(self))

        print("Initialized notMNIST. Length:", len(self))

    def __len__(self):
        return self.size or len(self.files) - self.misses

    def __getitem__(self, key):
        return self.ims[key],  self.labels[key]

notmnist_loader = None
"""
notmnist_data = NotMNIST()
sampler = sampler.BatchSampler(
    torch.utils.data.sampler.RandomSampler(notmnist_data),
    batch_size=batch_size,
    drop_last=False
)
notmnist_loader = DataLoader(
    notmnist_data,
    sampler=sampler
)
"""