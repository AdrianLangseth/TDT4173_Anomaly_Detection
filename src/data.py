import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import os
import random
from glob import glob
from PIL import Image

# TODO: Agree on batch size
batch_size = 256  # going higher than 43 results in NaN results if using SGD

# TODO: Do we want to normalize the data? We probably do. If so, agree on common parameters
normalize = False
if normalize:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])
else:
    transform = transforms.Compose([transforms.ToTensor()])

training_set_size = 50_000
test_set_size = 10_000
val_set_size = 10_000

# Training set is 60k ims, test set is 10k ims. 
# TODO: Agree on where validation set will come from and its size. I propose 10k from train
test_set, _ = torch.utils.data.random_split(
    torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform),
    [test_set_size, 10_000 - test_set_size]
)
train_set, val_set, _ = torch.utils.data.random_split(
    torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform),
    [training_set_size, val_set_size, 60_000 - (training_set_size + val_set_size)]
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)


notmnist_path = "../data/notMNIST_small/"
class NotMNIST:
    size = 100
    
    def set_dataset(self):
        files = random.sample(self.files, self.size)
        dataset_size = len(files)
        self.dataset = np.zeros((dataset_size, 28 * 28), dtype=np.float32)
        for i in range(dataset_size):
            self.dataset[i, :] = np.asarray(Image.open(files[i])).flatten()

        self.labels = -np.ones(batch_size)

    def __init__(self):
        self.files = [
            os.path.join(notmnist_path, directory, fn)
            for directory in os.listdir(notmnist_path)
            for fn in os.listdir(os.path.join(notmnist_path, directory))
        ]
        self.set_dataset()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        return (
            torch.tensor(self.dataset[key*batch_size : (key + 1)*batch_size, :]), 
            torch.from_numpy(self.labels[:min((key + 1)*batch_size, len(self))])
        )