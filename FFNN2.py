import numpy as np
import tensorflow.keras.activations as act
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM

# Dataset
def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']  # 60000 x 28 x 28, 600000
        x_test, y_test = f['x_test'], f['y_test']  # 10000 x 28 x 28, 100000
        return (x_train, y_train), (x_test, y_test)


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
