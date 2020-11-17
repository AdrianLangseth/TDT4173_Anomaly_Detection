import numpy as np
from PIL import Image
import os
from numpy.core._multiarray_umath import ndarray


def load_data(path) -> (tuple, tuple):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']  # 60000 x 28 x 28, 600000
        x_test, y_test = f['x_test'], f['y_test']  # 10000 x 28 x 28, 100000
        x_train, x_test = x_train/255.0, x_test/255.0
        return (x_train, y_train), (x_test, y_test)


def load_MNIST() -> (ndarray, ndarray, ndarray, ndarray):
    with np.load('mnist.npz') as f:
        x_train, y_train = f['x_train'], f['y_train']  # 60000 x 28 x 28, 600000
        x_test, y_test = f['x_test'], f['y_test']  # 10000 x 28 x 28, 100000
        x_train, x_test = x_train/255.0, x_test/255.0
        return x_train, y_train, x_test, y_test


def load_MNIST_subset(size: int) -> tuple:
    with np.load('mnist.npz') as f:
        x_train, y_train = f['x_train'][0:size], f['y_train'][0:size]
        x_test, y_test = f['x_test'], f['y_test']
        x_train, x_test = x_train/255.0, x_test/255.0
        return x_train, y_train, x_test, y_test


def create_nMNIST_dataset(size: int, path="notMNIST_small") -> ndarray:
    dataset = []

    for letter in os.listdir(path):
        if len(letter) == 1:
            for idx, imagename in enumerate(os.listdir(path + "/" + letter)):
                if idx == size:
                    break
                dataset.append(np.asarray(Image.open(path + "/" + letter + "/" + imagename)) / 255.0)
    return np.array(dataset)


def create_not_mnist_dataset():
    dataset = []
    for imagename in os.listdir("notMNIST_all"):
        if imagename == ".DS_Store":
            continue
        try:
            dataset.append(np.asarray(Image.open("notMNIST_all/" + imagename)) / 255.0)
        except (FileNotFoundError, OSError) as e:
            print(e)
    return np.array(dataset)


if __name__ == "__main__":
    x = create_not_mnist_dataset()
    print("hi")