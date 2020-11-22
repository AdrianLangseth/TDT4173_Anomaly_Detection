import torch
import numpy as np

from src.bnn.data import test_loader, notmnist_loader, MNISTData
from src.bnn.Net import prediction_data
from src.bnn.main import setup_model
from src.bnn.utils import get_loader_data

mnist_images, mnist_labels = MNISTData("test").data.tensors
notmnist_images, notmnist_labels = get_loader_data(notmnist_loader)


def get_prediction_data(train_set_i=0, dataset="test"):
    assert dataset in ("train", "val", "test", "notmnist")
    if dataset == "notmnist":
        ims, labs = get_loader_data(notmnist_loader)
    else:
        ims, labs = MNISTData(dataset).data[:]
    setup_model(train_set_i)
        
    return dict(zip(
        ("all_predictions", "entropies", "accuracy"),
        prediction_data(ims, labs)
    ))
    

if __name__ == '__main__':
    print(*get_prediction_data(train_set_i=4).items(), sep="\n", end="\n\n")
    print(*get_prediction_data(train_set_i=4, dataset="train").items(), sep="\n", end="\n\n")
    # print(*get_prediction_data(train_set_i=0).items(), sep="\n", end="\n\n")
    print(*get_prediction_data(train_set_i=0, dataset="notmnist").items(), sep="\n")
