import torch
import torchvision
import torchvision.transforms as transforms
import pyro

from data import train_loader, test_loader, val_loader

from BNet import BayesNet

pyro.clear_param_store()
net = BayesNet()
net.infer_parameters(train_loader)

total = 0.0
correct = 0.0
for images, labels in val_loader:
    pred = net.forward(images.view(-1, 784), n_samples=1)
    total += labels.size(0)
    correct += (pred.argmax(-1) == labels).sum().item()
print(f"Test accuracy: {correct / total * 100:.5f}")