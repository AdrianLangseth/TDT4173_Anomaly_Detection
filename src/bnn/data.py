import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split, sampler, TensorDataset
from torchvision.datasets.folder import ImageFolder
import numpy as np
import os
import itertools
from pathlib import Path
from PIL import Image


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

batch_size = 256
training_set_sizes = [50_000, 19_000, 7_000, 2_500, 1_000]
data_dir = os.path.join(Path(os.path.abspath(__file__)).parents[2], "data")
class MNISTData:
    train_data, val_data, test_data = None, None, None

    def __init__(self, mode, size=None):
        def transform(dataset):
            x = dataset.data.float().view(-1, 28*28) / 255.
            y = dataset.targets
            return TensorDataset(x.to(device), y.to(device))

        assert mode in ("train", "val", "test")
        if mode in ("train", "val"):
            if MNISTData.train_data is None or MNISTData.val_data is None:
                dataset = transform(datasets.MNIST(root=data_dir, train=True, download=True))
                MNISTData.train_data, MNISTData.val_data = random_split(dataset, [size or 50_000, 10_000])
        elif mode == "test":
            if MNISTData.test_data is None:
                MNISTData.test_data = transform(datasets.MNIST(root=data_dir, train=False, download=True))

        self.data = getattr(MNISTData, mode + "_data")
        self.size = size if size is not None else len(self.data)

    def __len__(self):
        return getattr(self, "size", len(self.data))
    
    def __getitem__(self, key):
        return self.data[key]

    def loader(self):
        return DataLoader(self, batch_size=batch_size, shuffle=True)


def get_im_data(path):
    with open(path, 'rb') as f:
        return np.array(Image.open(f), dtype=np.float32).flatten()

notmnist_path = os.path.join(data_dir, "notMNIST_small")
class NotMNISTData(ImageFolder):
    size = 10_000

    def __init__(self):
        super().__init__(notmnist_path)
        self.data = torch.stack([
           torch.tensor(get_im_data(path))
           for path, _ in self.samples
        ]) / 255.
        self.targets = torch.tensor(self.targets, dtype=torch.int8) + ord("A")
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        return self.data[key], self.targets[key]


test_loader = MNISTData("test").loader()
val_loader =  MNISTData("val").loader()
train_loader = None

def set_train_size(training_set_size):
    global train_loader
    train_loader = MNISTData("train", size=training_set_size).loader()

notmnist_loader = DataLoader(
    NotMNISTData(),
    batch_size=batch_size, 
    shuffle=True
)
