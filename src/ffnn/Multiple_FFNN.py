import numpy as np
import tensorflow.keras.activations as act
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import tensorflow.keras.optimizers as opt
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from data_load import load_MNIST_subset, load_MNIST_validation_data


def create_model(size: int, is_drop_out: bool) -> None:
    if is_drop_out is True:
        # This is the model *with* drop-out
        drop_out_rate = .269
        model_location = 'dropout_models'
        text_message = "ON "

    else:
        # This is the plain one
        drop_out_rate = .0
        model_location = 'ffnn_models'
        text_message = "OFF"

    # Get data
    x_train, y_train, x_test, y_test = load_MNIST_subset(size)
    x_val, y_val = load_MNIST_validation_data()

    # Set batch-size
    bs = int(np.ceil(2 ** (np.round(np.log2(size / 500)))))

    # If model exists, we can load it from file
    # --> If we want to re-learn it, we need to delete the file first

    try:
        # Trying to load it
        model = KM.load_model(f'./{model_location}/model_{size}')
        model.compile(optimizer=opt.Adam(0.001),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

    except IOError:
        # Not there....
        inputs = KL.Input(shape=(28, 28))
        layer_result = KL.Flatten()(inputs)
        layer_result = KL.Dense(512, activation=act.sigmoid)(layer_result)
        if is_drop_out is True:
            # Dropout on hidden layer, dropout also in test-phase. Enforced by setting "learning=True"
            layer_result = KL.Dropout(rate=drop_out_rate)(layer_result, training=True)

        outputs = KL.Dense(10, activation=act.softmax)(layer_result)
        model = KM.Model(inputs, outputs)
        model.compile(
            optimizer=opt.Adam(0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # model.summary()

        tb_callback = TensorBoard(
            log_dir=f'./logs/{model_location}/{size}',
            histogram_freq=0,
            write_graph=True,
            write_images=True,
        )
        es_callback = EarlyStopping(
            monitor='val_loss',
            patience=15,
        )

        model.fit(
            x_train, y_train,
            batch_size=bs,
            epochs=1000,
            verbose=0,
            callbacks=[tb_callback, es_callback],
            validation_data=(x_val, y_val),
        )
        model.save(f'./{model_location}/model_{size}')

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    print(
        f"Size {size:7d}, batch-size {bs:4d}, drop-off {text_message}: "
        f"test_acc = {test_acc:.4f}, test_loss = {test_loss:8.6f}"
    )

def create_FFNN_Model(size: int) -> None:
    # Don't repeat code. This function is now a wrapper for create_model, where the real work goes on.
    # Since this method used to create the model **without** the dropout-layer, the call used is with
    # is_drop_out=False.

    # Keeping the method here to make the interface simpler: create_FFNN_Model works as before

    '''
    x_train, y_train, x_test, y_test = load_MNIST_subset(size)
    x_val, y_val = load_MNIST_validation_data()

    # Model
    bs = int(np.ceil(2 ** (np.round(np.log2(size / 500)))))

    try:
        model = KM.load_model(f'./ffnn_models/model_{size}')
        model.compile(optimizer=opt.Adam(0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    except IOError:
        inputs = KL.Input(shape=(28, 28))
        l = KL.Flatten()(inputs)
        l = KL.Dense(512, activation=act.sigmoid)(l)
        outputs = KL.Dense(10, activation=act.softmax)(l)

        model = KM.Model(inputs, outputs)
        # model.summary()
        tbCallBack = TensorBoard(log_dir='./logs/ffnn/' + str(size), histogram_freq=0,
                                 write_graph=True, write_images=True)
        callback = EarlyStopping(monitor='val_loss', patience=15)
        model.compile(optimizer=opt.Adam(0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        model.fit(x_train, y_train,
                  batch_size=bs,
                  epochs=1000,
                  verbose=0,
                  callbacks=[tbCallBack, callback],
                  validation_data=(x_val, y_val),
                  )
        model.save("ffnn_models/model_" + str(size))

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    """
    predictions = model.predict(x_test)
    # np.savetxt(fname="./Multiple_FFNN__create_FFNN_Model.txt", X=predictions, fmt='%.6f')
    # np.savetxt(fname="./Multiple_FFNN__class.txt", X=np.argmax(predictions, axis=1), fmt='%.d')
    correctness = np.equal(np.argmax(predictions, axis=1), y_test)
    own_acc = np.mean(correctness)
    assert (np.isclose(own_acc, test_acc))
    """

    print(f"Size {size:7d}, batch-size {bs:4d}, test_acc = {test_acc:.4f}, test_loss = {test_loss:8.6f}")
    '''

    create_model(size=size, is_drop_out=False)


def old_create_FFNN_Model(size:int):
    x_train, y_train, x_test, y_test = load_MNIST_subset(size)
    x_train, x_test = x_train/255.0, x_test/255.0

    # Model
    inputs = KL.Input(shape=(28, 28))
    l = KL.Flatten()(inputs)
    l = KL.Dense(512, activation=act.sigmoid)(l)
    outputs = KL.Dense(10, activation=act.softmax)(l)

    model = KM.Model(inputs, outputs)
    # model.summary()
    # tbCallBack = TensorBoard(log_dir='./logs/ffnn/' + str(size), histogram_freq=0, write_graph=True, write_images=True)
    callback = EarlyStopping(monitor='loss', patience=15)
    model.compile(optimizer=opt.Adam(0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    bs = int(np.ceil(2**(np.round(np.log2(size/500)))))
    model.fit(x_train, y_train,
              batch_size=bs,
              epochs=1000,
              verbose=1,
              callbacks=[callback]
              )

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(size, ":", test_acc, test_loss)
    model.save("ffnn_models/model_" + str(size))


if __name__ == '__main__':
    needed_model_sizes = [1000, 2500, 7000, 19000, 50000]
    for size in needed_model_sizes:
        create_FFNN_Model(size)
