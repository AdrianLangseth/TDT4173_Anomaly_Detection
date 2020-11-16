import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, sampler

import numpy as np
import os
import random
from glob import glob
from PIL import Image

training_set_sizes = [50_000, 19_000, 7_000, 2_500, 1_000]
training_set_index = 0
batch_size = 256  # going higher than 43 results in NaN results if using SGD

normalize = True
if normalize:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.0, ), (1.0, ))
    ])
else:
    transform = transforms.Compose([transforms.ToTensor()])

training_set_size = training_set_sizes[training_set_index]
val_set_size = 10_000
test_set_size = 10_000

test_set, _ = random_split(
    torchvision.datasets.MNIST(root='../../data', train=False, download=True, transform=transform),
    [test_set_size, 10_000 - test_set_size]
)
train_set, val_set, _ = random_split(
    torchvision.datasets.MNIST(root='../../data', train=True, download=True, transform=transform),
    [training_set_size, val_set_size, 60_000 - (training_set_size + val_set_size)]
)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)


notmnist_path = "../../data/notMNIST_small/"
class NotMNIST:
    misses = 0
    size = 10_000

    def __size(self):
        return self.size or len(self.files)

    def __setup_ims(self):
        self.ims = np.zeros((self.__size(), 28, 28), dtype=np.float32)
        self.labels = np.zeros(self.__size(), dtype=np.int8)
        for i, fn in enumerate(self.files):
            try:
                self.ims[i - self.misses, :, :] = np.asarray(Image.open(fn))
                self.labels[i - self.misses] = ord(fn.replace("\\", "/").split("/")[-2]) # ascii value of letter
            except:
                self.misses += 1
            if i - self.misses >= self.__size():
                break
    
    def __init__(self):
        self.files = [
            os.path.join(notmnist_path, directory, fn)
            for directory in os.listdir(notmnist_path)
            for fn in os.listdir(os.path.join(notmnist_path, directory))
        ]

    def __len__(self):
        if self.__dict__.get("ims") is None:
            self.__setup_ims()
        return len(self.files) - self.misses

    def __getitem__(self, key):
        if self.__dict__.get("ims") is None:
            self.__setup_ims()
        return self.ims[key],  self.labels[key]

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