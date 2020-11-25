from src.ffnn.data_load import load_MNIST, get_high_entropy_mnist_test
from src.ffnn.FFNN_predictor import do_MNIST_entropy, model_predictor
import numpy as np

def get_entropy_of_maxes():
    """

    :return: {y:[7], model: [(predicted, confidence)]}
    """
    high_entropy_list = get_high_entropy_mnist_test()  # [(image1, fasit1), (image1, fasit2)]

    d = {}

    images = []
    values = []

    for i in high_entropy_list:
        images.append(i[0])
        values.append(i[1])

    d["y"] = np.array(values)
    d["d"] = []
    d["f"] = []

    entropy = do_MNIST_entropy(np.array(images))

    model_paths = ["ffnn_models", "dropout_models"]

    for model in model_paths:
        pred = model_predictor(model + "/model_50000", np.array(images), np.array(values))[0]
        for i in pred:
            d[model[0]].append((np.argmax(i), i[np.argmax(i)]))

    return d

if __name__ == "__main__":
    _, _, x_test, y_test = load_MNIST()
    x = get_entropy_of_maxes()
    print(x)
