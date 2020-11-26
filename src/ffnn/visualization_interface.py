from src.ffnn.notMNIST import test_notmnist_entropy_for_all
from src.ffnn.FFNN_predictor import get_all_predictions, train_MNIST_entropy, test_MNIST_entropy
import time
from src.ffnn.webcode import get_entropy_of_maxes


def entropy_not_mnist() -> dict:
    """
    A function whose sole function is to be a convenience wrapper for the visualization creator of not_mnist entropy
    getters.
    :return:
    """
    return test_notmnist_entropy_for_all()


def entropy_mnist_train() -> dict:
    """
    A function whose sole function is to be a convenience wrapper for the visualization creator of mnist_train entropy
    getters.
    :return: Dictionary of all the entropy of all the mnist_train predictions of all models.
    """
    return train_MNIST_entropy()


def entropy_mnist_test() -> dict:
    """
    A function whose sole function is to be a convenience wrapper for the visualization creator of mnist_test entropy
    getters.
    :return: Dictionary of all the entropy of all the mnist_test predictions of all models.
    """
    return test_MNIST_entropy()


def mnist() -> dict:
    """
    A function whose sole function is to be a convenience wrapper for the visualization creator of mnist predictions
    getters.
    :return: Dictionary of all the mnist predictions of all models, as well as the actual y_value.
    """
    return get_all_predictions()


def webcode() -> dict:
    """
    A function whose sole function is to be a convenience wrapper for the website creator of high_entropy_mnist images.
    :return: Dictionary whose keys are model indicator, the values is a list of tuples holding
    (predicted value, confidence).
    """
    return get_entropy_of_maxes()


if __name__ == "__main__":
    t0 = time.time()
    train = entropy_mnist_train()
    t1 = time.time()
    print("Entropi train: " + str(t1-t0))
    test = entropy_mnist_test()
    t2 = time.time()
    print("Entropi test: " + str(t2-t1))
    notmnist = entropy_not_mnist()
    t3 = time.time()
    print("Entropi not_mnist: " + str(t3-t2))
    pred = mnist()
    t4 = time.time()
    print("Predictions: " + str(t4-t3))
    preds = webcode()
    t5 = time.time()
    print("Total: " + str(t5-t0))
