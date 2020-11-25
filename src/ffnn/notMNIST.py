import numpy as np
import tensorflow.keras.models as KM
from src.ffnn.data_load import create_not_mnist_doubleset
import scipy.stats as stats
import tensorflow as tf
import hyperas
import scipy
import hyperopt


def test_notmnist_entropy_for_all() -> dict:
    """
    predicts not_mnist dataset for all models and finds entropy of each prediction.
    :return: dictionary of each model and the entropy of each prediction.
    """
    x_notmnist, fasit = create_not_mnist_doubleset()
    dropout_runs = 100
    x_notmnist, fasit = x_notmnist[:10000], fasit[:10000]
    d = {"y": fasit}
    model_paths = ["ffnn_models", "dropout_models"]
    sizes = [1000, 2500, 7000, 19000, 50000]
    for folder in model_paths:
        for size in sizes:
            path = folder + "/model_" + str(size)
            model = KM.load_model(path)

            if folder == "dropout_models":
                pred = np.zeros((len(x_notmnist), 10))
                for i in range(dropout_runs):
                    temp_predict = model.predict(x_notmnist)
                    for idx, n in enumerate(np.argmax(temp_predict, axis=1)):
                        pred[idx][n] += 1
                pred = pred / dropout_runs
            else:
                pred = model.predict(x_notmnist)
            d[folder[0] + str(size)] = stats.entropy(pred, axis=1)
    return d


if __name__ == "__main__":
    print("The models predicted are: ", test_notmnist_entropy_for_all().keys())