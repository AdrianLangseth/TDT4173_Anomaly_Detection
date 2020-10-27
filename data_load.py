import numpy as np


def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']  # 60000 x 28 x 28, 600000
        x_test, y_test = f['x_test'], f['y_test']  # 10000 x 28 x 28, 100000
        return (x_train, y_train), (x_test, y_test)
