import data_load

from hyperas import optim
from hyperas.distributions import uniform, choice
from hyperopt import Trials, STATUS_OK, tpe

import tensorflow.keras.layers as KL
import tensorflow.python.keras.models as KM
import tensorflow.keras.activations as act
import tensorflow.keras.optimizers as opt

import numpy as np


def create_model(x_train, y_train, x_test, y_test):
    inputs = KL.Input(shape=(28, 28))
    l = KL.Flatten()(inputs)
    l = KL.Dropout(0)(l)  # Dropout on Visible layer
    l = KL.Dense(512, activation={{choice(["relu", "sigmoid", "elu"])}})(l) #
    l = KL.Dropout({{uniform(0, 0.5)}})(l)  # Dropout on hidden layer
    outputs = KL.Dense(10, activation={{choice(["softmax", "sigmoid"])}})(l)  #

    model = KM.Model(inputs, outputs)

    model.compile(optimizer=opt.Adam(learning_rate={{uniform(0, 0.02)}}), loss="sparse_categorical_crossentropy", metrics=["accuracy"])  # choice(["sgd", "adam", "RMSprop"])}}
    result = model.fit(x_train, y_train,
                       epochs={{choice([3, 5, 10])}},
                       batch_size={{choice([32, 64, 128])}},
                       verbose=0,
                       validation_split=(10000 / len(x_train)))

    highest_val_accuracy = np.amax(result.history['val_accuracy'])
    print('Highest validation accuracy of epoch:', highest_val_accuracy)
    return {'loss': -highest_val_accuracy, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data_load.load_MNIST,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data_load.load_MNIST()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
