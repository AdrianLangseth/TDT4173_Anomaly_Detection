import numpy as np
import tensorflow.keras.activations as act
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import tensorflow.python.keras

from data_load import load_data

(x_train, y_train), (x_test, y_test) = load_data('mnist.npz')
x_train, x_test = x_train/255.0, x_test/255.0

# Model
inputs = KL.Input(shape=(28, 28))
l = KL.Flatten()(inputs)
l = KL.Dense(512, activation=act.relu)(l)
outputs = KL.Dense(10, activation=act.softmax)(l)


model = KM.Model(inputs, outputs)
model.summary()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=10)
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Loss: {0} - Test Acc: {1}".format(test_loss, test_acc))

model.save("ffnn_model")

