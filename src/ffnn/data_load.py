import numpy as np
from PIL import Image
import os
from numpy.core._multiarray_umath import ndarray

######### DEPRECATED ########
def load_data(path) -> (tuple, tuple):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']  # 60000 x 28 x 28, 600000
        x_test, y_test = f['x_test'], f['y_test']  # 10000 x 28 x 28, 100000

    scale = float(np.max(x_train))
    x_train, x_test = x_train / scale, x_test / scale
    return (x_train, y_train), (x_test, y_test)


def load_MNIST() -> (ndarray, ndarray, ndarray, ndarray):

    # This one does the same as load_data, but pre-assumes the location of the file,
    # and unpacks the two tuples returned from load_data into a elements
    '''
    with np.load('mnist.npz') as f:
        x_train, y_train = f['x_train'], f['y_train']  # 60000 x 28 x 28, 600000
        x_test, y_test = f['x_test'], f['y_test']  # 10000 x 28 x 28, 100000

    scale = float(np.max(x_train))
    x_train, x_test = x_train/scale, x_test/scale
    '''

    (x_train, y_train), (x_test, y_test) = load_data('mnist.npz')

    return x_train, y_train, x_test, y_test


def load_MNIST_validation_data() -> (ndarray, ndarray):
    # Returns the last 10.000 examples from the training data. This will play the role as VALIDATION
    x_train, y_train, x_test, y_test = load_MNIST()
    return x_train[-10000:], y_train[-10000:]


######### DEPRECATED ########
def load_MNIST_subset(size: int) -> (ndarray, ndarray, ndarray, ndarray):

    # Again, redoing the same code. Not needed, and cause for problems. AVoid it

    '''
    with np.load('mnist.npz') as f:
        x_train, y_train = f['x_train'][0:size], f['y_train'][0:size]
        x_test, y_test = f['x_test'], f['y_test']

    scale = float(np.max(x_train))
    x_train, x_test = x_train/scale, x_test/scale
    return x_train, y_train, x_test, y_test
    '''
    x_train, y_train, x_test, y_test = load_MNIST()
    x_train, y_train = x_train[:size], y_train[:size]

    return x_train, y_train, x_test, y_test


######### DEPRECATED ########
def create_nMNIST_dataset(size: int, path="notMNIST_small") -> ndarray:
    dataset = []

    for letter in os.listdir(path):
        if len(letter) == 1:
            for idx, imagename in enumerate(os.listdir(path + "/" + letter)):
                if idx == size:
                    break
                dataset.append(np.asarray(Image.open(path + "/" + letter + "/" + imagename)) / 255.0)
    return np.array(dataset)


######### DEPRECATED ########
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


######### DEPRECATED ########
def create_not_mnist_dict():
    dataset = {}
    for imagename in os.listdir("notMNIST_all"):
        if imagename == ".DS_Store":
            continue
        try:
            if ord(imagename[0]) in dataset.keys():
                dataset[ord(imagename[0])].append(np.asarray(Image.open("notMNIST_all/" + imagename)) / 255.0)
            else:
                dataset[ord(imagename[0])] = [(np.asarray(Image.open("notMNIST_all/" + imagename)) / 255.0)]
        except (FileNotFoundError, OSError) as e:
            print(e)
    return np.array(dataset)


def back_create_not_mnist_doubleset():
    x = []
    y = []
    for imagename in os.listdir("notMNIST_all"):
        if imagename == ".DS_Store":
            continue
        try:
            x.append(np.asarray(Image.open("notMNIST_all/" + imagename)) / 255.0)
            y.append(ord(imagename[0]))
        except (FileNotFoundError, OSError) as e:
            print(e)
    return np.array(x), np.array(y)


def create_not_mnist_doubleset():

    try:
        with np.load("./notMNIST_all/all_data.npz") as f:
            x, y = f['x'], f['y']

    except FileNotFoundError:
        # Have to parse the image files, as the .npz numpy file is not there

        x = []
        y = []
        for image_name in os.listdir("notMNIST_all"):
            if image_name == ".DS_Store":
                continue
            try:
                image_as_array = np.asarray(Image.open("notMNIST_all/" + image_name))
                scale = float(np.max(image_as_array))

                # Scale data so the input is in range [0, 1]
                # and the class is in the range [0, 1, .., no_classes - 1]
                x.append(image_as_array / scale)
                y.append(ord(image_name[0]) - ord("A"))

            except (FileNotFoundError, OSError) as e:
                print(f"Skipping the file {image_name}, as it gave error {e}")

        x, y = np.array(x), np.array(y)
        np.savez(file="./notMNIST_all/all_data.npz", x=x, y=y)
    return x, y


if __name__ == "__main__":
    mnist_x_subset, mnist_y_subset, x_mnist_test, y_mnist_test = load_MNIST_subset(size=500)
    not_mnist_x, not_mnist_y = create_not_mnist_doubleset()
    not_mnist_dict = create_not_mnist_dict()
