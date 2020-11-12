from PIL import Image
import numpy as np
import tensorflow.keras as KK
from data_load import load_MNIST_subset
import os
import scipy.stats as stats
from numpy.core._multiarray_umath import ndarray
import matplotlib.pyplot as plt


def create_nMNIST_dataset(size: int, path="notMNIST_small"):
    dataset = []

    for letter in os.listdir(path):
        if len(letter) == 1:
            for idx, imagename in enumerate(os.listdir(path + "/" + letter)):
                if idx == size:
                    break
                dataset.append(np.asarray(Image.open(path + "/" + letter + "/" + imagename)) / 255.0)
    return np.array(dataset)


# x = create_nMNIST_dataset(1)
# model = KK.models.load_model("ffnn_models/model_50000")
# y = model.predict(x)
# maxes = [(max(lists), np.argmax(lists)) for lists in y]
# print(maxes)


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

def test_notmnist_entropy_for_all(size: int) -> (ndarray, ndarray):
    x_notmnist = create_nMNIST_dataset(size)
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
# mnist, nmnist = test_notmnist_entropy(10)


d = test_notmnist_entropy_for_all(5)
fig, ax = plt.subplots()

mods = d.keys()
vals = d.values()

for mod in mods:
    ax.scatter([mod for i in range(len(d[mod]))], d[mod])
plt.show()

# typ = ["NOT_MNIST", "MNIST"]
# vals = [nmnist[0], mnist[0]]
#
# y = nmnist[0].tolist()
# yy = mnist[0].tolist()
#
# fig, ax = plt.subplots()
# ax.scatter(["NOT_MNIST" for i in range(len(y))], y)
# ax.scatter(["MNIST" for j in range(len(yy))], yy)
#
# plt.savefig("hei.jpg")


# def plot_entropy()

# im = Image.open("notMNIST_small/A/MDEtMDEtMDAudHRm.png")
#
# print(im.format)
# print(im.size)
# print(im.mode)
#
# ar = [np.asarray(im)/255.0]
# ar = np.array(ar)
#
# x_train, _, _, _ = load_MNIST_subset(5)
#
# model = KK.models.load_model("ffnn_models/model_50000")
# print(model.predict(ar))