from PIL import Image
import numpy as np
import tensorflow.keras as KK
from data_load import load_MNIST_subset, create_not_mnist_dataset, create_nMNIST_dataset
import os
import scipy.stats as stats
from numpy.core._multiarray_umath import ndarray
import matplotlib.pyplot as plt
import seaborn as sns






def test_model_nMNIST(size: int, model_path: str, threshold: float):
    x_notmnist = create_nMNIST_dataset(size)
    model = KK.models.load_model(model_path)
    pred = model.predict(x_notmnist)
    return np.average([(max(sm_out) < threshold) for sm_out in pred])


def test_notmnist_entropy(size: int, model_path="ffnn_models/model_1000") -> (ndarray, ndarray):
    x_notmnist = create_nMNIST_dataset(size)
    x_train, y_train, x_test, y_test = load_MNIST_subset(size)
    model = KK.models.load_model(model_path)
    not_pred = model.predict(x_notmnist)
    pred = model.predict(x_test[0:size*10])
    not_entropy = stats.entropy(not_pred, axis=1)
    entropy = stats.entropy(pred, axis=1)
    return (entropy, pred), (not_entropy, not_pred)


def test_notmnist_entropy_for_all() -> dict:
    x_notmnist = create_not_mnist_dataset()
    d = {}
    model_paths = ["ffnn_models", "dropout_models"]
    sizes = [1000, 2500, 7000, 19000, 50000]
    for folder in model_paths:
        for size in sizes:
            path = folder + "/model_" + str(size)
            model = KK.models.load_model(path)
            pred = model.predict(x_notmnist)
            not_entropy = stats.entropy(pred, axis=1)
            d[folder[0] + str(size)] = not_entropy
    return d


# print(test_model_nMNIST(100, "ffnn_models/model_50000", 0.8))
#mnist, nmnist = test_notmnist_entropy(10)

