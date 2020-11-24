import numpy as np
import tensorflow.keras.activations as act
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import tensorflow.keras.optimizers as opt


from data_load import load_MNIST

x_train, y_train, x_test, y_test = load_MNIST()
x_train, x_test = x_train/255.0, x_test/255.0

# Model
inputs = KL.Input(shape=(28, 28))
l = KL.Flatten()(inputs)
l = KL.Dense(512, activation=act.sigmoid)(l)
outputs = KL.Dense(10, activation=act.softmax)(l)


model = KM.Model(inputs, outputs)
model.summary()
model.compile(optimizer=opt.Adam(0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1)
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Loss: {0} - Test Acc: {1}".format(test_loss, test_acc))


# model.save("ffnn_model")


######### THIS ENTIRE FILE IS DEPRECATED AND WILL BE REMOVED ########
