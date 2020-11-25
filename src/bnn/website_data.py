import torch
import os
import sys

from Net import prediction_data
from main import setup_model
from settings import SRC_DIR, device

sys.path.append(os.path.join(SRC_DIR, 'ffnn'))  # make ffnn files available as imports
from data_load import get_high_entropy_mnist_test


setup_model()


def get_high_ffnn_entropy_instances_entropies():
    x, y = zip(*get_high_entropy_mnist_test())
    x = torch.tensor(x, dtype=torch.float32).to(device).view(-1, 28*28)
    y = torch.tensor(y).to(device)
    return prediction_data(x, y)[1]


if __name__ == '__main__':
    print(
        get_high_ffnn_entropy_instances_entropies()
    )