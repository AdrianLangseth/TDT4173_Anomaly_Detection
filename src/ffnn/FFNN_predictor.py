import numpy as np
import tensorflow.keras as KK
from numpy.core._multiarray_umath import ndarray
from scipy.stats import entropy
from data_load import load_MNIST, create_not_mnist_doubleset


def model_predictor(model_repo_path: str, x_test_values: ndarray, y_test_values: ndarray) -> (ndarray, ndarray):
    model = KK.models.load_model(model_repo_path)
    dropout_runs = 100
    if model_repo_path.split("/")[0] == "dropout_models":

        pred = np.zeros((len(x_test_values), 10))

        for i in range(dropout_runs):
            temp_predict = model.predict(x_test_values)

            # Quicker version -- avoids loop over all elements in the test-set
            pred[np.arange(pred.shape[0]), np.argmax(temp_predict, axis=1)] += 1
            """
            for idx, n in enumerate(np.argmax(temp_predict, axis=1)):
                pred[idx][n] += 1
            """

        predictions = pred / dropout_runs

    else:
        predictions = model.predict(x_test_values)
    return predictions, y_test_values


def get_int_predictions(model_repo_path: str, data_load_function=load_MNIST) -> (ndarray, ndarray):
    _, _, x_test, y_test = data_load_function()
    predicted, correct = model_predictor(model_repo_path, x_test, y_test)
    return np.argmax(predicted, axis=1), correct


def test_nMNIST_prediction(model_repo_path: str, data: ndarray):
    model = KK.models.load_model(model_repo_path)
    return model.predict(data)


def train_MNIST_entropy():
    x_train, _, _, _ = load_MNIST()
    x_train = x_train[:10000]
    return do_MNIST_entropy(input_data=x_train)

def do_MNIST_entropy(input_data):

    model_paths = ["ffnn_models", "dropout_models"]
    dropout_runs = 100
    d = {}
    sizes = [1000, 2500, 7000, 19000, 50000]
    for folder in model_paths:
        for size in sizes:
            path = folder + "/model_" + str(size)
            model = KK.models.load_model(path)
            if folder == "dropout_models":
                pred = np.zeros((len(input_data), 10))
                for i in range(dropout_runs):
                    temp_predict = model.predict(input_data)

                    # Should average out predictions at "output level" -- prior to argmax.
                    # Doing it *after* argmax suppresses a situation where, e.g., the model
                    # has a 60% chance of class0 and 40% for class1 in all realizations of the dropouts.
                    pred += temp_predict
                    """
                    for idx, n in enumerate(np.argmax(temp_predict, axis=1)):
                        pred[idx][n] += 1
                    """

                pred = pred / dropout_runs
            else:
                pred = model.predict(input_data)
            model_entropy = entropy(pred, axis=1)
            d[folder[0] + str(size)] = model_entropy
    return d

def not_MNIST_entropy(no_random_images=10000):
    x, _ = create_not_mnist_doubleset()
    chooser = np.random.permutation(x.shape[0])[:no_random_images]
    x = x[chooser, :]
    return do_MNIST_entropy(input_data=x)


def test_MNIST_entropy():
    _, _, x_test, _ = load_MNIST()
    return do_MNIST_entropy(input_data=x_test)


def get_all_predictions():
    all_results = {}
    model_paths = ["ffnn_models", "dropout_models"]
    sizes = [1000, 2500, 7000, 19000, 50000]
    pred, correct = None, None
    for folder in model_paths:
        for size in sizes:
            path = folder + "/model_" + str(size)
            if len(all_results) == 0:
                pred, correct = get_int_predictions(folder + "/model_" + str(size), load_MNIST)
                all_results["y"] = correct
                all_results[path] = pred
            else:
                all_results[path] = get_int_predictions(folder + "/model_" + str(size), load_MNIST)[0]

            print(f"Test-set accuracy = {np.mean(np.equal(all_results[path] , correct)):.4f} for model {path}")
    return all_results


if __name__ == '__main__':

    # Test prediction quality (accuracy) of the model. Dropout model averages over some runs.
    print(f"Calculating test-set accuracy for all models:\n{100 * '='}")
    get_all_predictions()

    # Test entropy-part; looping over the three test-scenarios, then dumping results:
    functions = [train_MNIST_entropy, test_MNIST_entropy, not_MNIST_entropy]
    descriptions = ['MNIST training-data', 'MNIST test-data', 'notMNIST data']

    for func, desc in zip(functions, descriptions):
        result = func()
        print(f"\n\nRunning entropy-calculations on {desc}:\n{100*'='}")
        for key in result.keys():
            print(f"Model {key:7s} gave mean entropy {np.mean(result[key]):8.4f}")

