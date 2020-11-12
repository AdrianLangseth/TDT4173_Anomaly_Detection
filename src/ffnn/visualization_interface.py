from src.ffnn.notMNIST import test_notmnist_entropy_for_all
from src.ffnn.FFNN_predictor import get_all_predictions


def not_mnist() -> dict:
    """
    A function whose sole function is to be a convenience wrapper for the visualization creator of not_mnist entropy
    getters.
    :return:
    """
    return test_notmnist_entropy_for_all()


def mnist() -> dict:
    """
    A function whose sole function is to be a convenience wrapper for the visualization creator of mnist predictions
    getters.
    :return: Dictionary of all the mnist predictions of all models, as well as the actual y_value.
    """
    return get_all_predictions()