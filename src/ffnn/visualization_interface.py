from src.ffnn.notMNIST import test_notmnist_entropy_for_all
from src.ffnn.FFNN_predictor import get_all_predictions, train_MNIST_entropy, test_MNIST_entropy


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


if __name__ == "__main__":
    train = entropy_mnist_train()
    test = entropy_mnist_test()
    notmnist = entropy_not_mnist()
    pred = mnist()
    pass
