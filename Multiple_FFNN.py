import numpy as np
import tensorflow.keras.activations as act
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import tensorflow.keras.optimizers as opt
from time import sleep

from data_load import load_MNIST_subset


def create_FFNN_Model(size:int):
    x_train, y_train, x_test, y_test = load_MNIST_subset(size)
    x_train, x_test = x_train/255.0, x_test/255.0

    # Model
    inputs = KL.Input(shape=(28, 28))
    l = KL.Flatten()(inputs)
    l = KL.Dense(512, activation=act.sigmoid)(l)
    outputs = KL.Dense(10, activation=act.softmax)(l)


    model = KM.Model(inputs, outputs)
    # model.summary()
    model.compile(optimizer=opt.Adam(0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    bs = int(np.ceil(2**(np.round(np.log2(size/500)))))
    model.fit(x_train, y_train, batch_size=bs, epochs=10, verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(size, ":", bs, test_acc)
    model.save("ffnn_models/model_" + str(size))


if __name__ == '__main__':
    needed_model_sizes = [1000, 2500, 7000, 19000, 50000]
    for size in needed_model_sizes:
        create_FFNN_Model(size)
