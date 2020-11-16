import numpy as np
import tensorflow.keras as KK
from numpy.core._multiarray_umath import ndarray
from scipy.stats import entropy
from src.ffnn.data_load import load_MNIST


def model_predictor(model_repo_path: str, x_test_values: ndarray, y_test_values: ndarray) -> (ndarray, ndarray):
    model = KK.models.load_model(model_repo_path)
    predictions = model.predict(x_test_values)
    return predictions, y_test_values


def get_int_predictions(model_repo_path: str, data_load_function=load_MNIST) -> (ndarray, ndarray):
    _, _, x_test, y_test = data_load_function()
    predicted, fasit = model_predictor(model_repo_path, x_test, y_test)
    return np.argmax(predicted, axis=1), fasit


def test_nMNIST_prediction(model_repo_path: str, data: ndarray):
    model = KK.models.load_model(model_repo_path)
    return model.predict(data)


def train_MNIST_entropy():
    x, y, _, _ = load_MNIST()
    model_paths = ["ffnn_models", "dropout_models"]
    d = {}
    sizes = [1000, 2500, 7000, 19000, 50000]
    for folder in model_paths:
        for size in sizes:
            path = folder + "/model_" + str(size)
            model = KK.models.load_model(path)
            pred = model.predict(x[:size])
            model_entropy = entropy(pred, axis=1)
            d[folder[0] + str(size)] = model_entropy
    return d


def test_MNIST_entropy():
    _, _, x_test, y_test = load_MNIST()
    model_paths = ["ffnn_models", "dropout_models"]
    d = {}
    sizes = [1000, 2500, 7000, 19000, 50000]
    for folder in model_paths:
        for size in sizes:
            path = folder + "/model_" + str(size)
            model = KK.models.load_model(path)
            pred = model.predict(x_test)
            model_entropy = entropy(pred, axis=1)
            d[folder[0] + str(size)] = model_entropy
    return d


def get_all_predictions():
    all = {}
    model_paths = ["ffnn_models", "dropout_models"]
    sizes = [1000, 2500, 7000, 19000, 50000]
    for folder in model_paths:
        for size in sizes:
            path = folder + "/model_" + str(size)
            if len(all) == 0:
                pred, fasit = get_int_predictions(folder + "/model_" + str(size), load_MNIST)
                all["y"] = fasit
                all[path] = pred
            else:
                all[path] = get_int_predictions(folder + "/model_" + str(size), load_MNIST)[0]
    return all


if __name__ == '__main__':
    # x_train, y_train, x_test, y_test = load_MNIST()
    # pred, y = model_predictor('ffnn_models/model_1000', x_test[0:10], y_test[0:10])
    # print(np.argmax(pred, axis=1))
    # print(y)
    # pred, fasit = model_predictor('ffnn_models/model_50000')
    # x = np.equal(pred, fasit)
    # print(sum(x)/len(pred))
    # get_int_predictions("ffnn_models/model_50000", load_MNIST)
    # x = test_MNIST_entropy()
    # print(2)
    pass
