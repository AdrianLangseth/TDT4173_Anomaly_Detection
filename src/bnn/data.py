import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, sampler
from torchvision.datasets.folder import ImageFolder, default_loader
import numpy as np
import os
from PIL import Image


batch_size = 256
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.0, ), (1.0, ))
])
test_set = torchvision.datasets.MNIST(root='../../data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
train_loader, val_loader = None, None


def setup_train_val_loaders(training_set_size, val_set_size=10_000):
    train_set, val_set, _ = random_split(
        torchvision.datasets.MNIST(root='../../data', train=True, download=True, transform=transform),
        [training_set_size, val_set_size, 60_000 - (training_set_size + val_set_size)]
    )
    global train_loader, val_loader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)


notmnist_path = "../../data/notMNIST_small/"
class NotMNISTFolder(ImageFolder):
    size = 10_000

    def __init__(self):
        super().__init__(notmnist_path)
        self.data = torch.cat(tuple(
            transform(Image.open(path)) for path, _ in self.samples
        ))
        self.targets = np.array(self.targets, dtype=np.int8) + ord("A")

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        return self.data[key], self.targets[key]


notmnist_data = NotMNISTFolder()
notmnist_set = sampler.BatchSampler(
    sampler.RandomSampler(notmnist_data),
    batch_size=batch_size,
    drop_last=False
)
notmnist_loader = DataLoader(
    notmnist_data,
    sampler=notmnist_set
)
