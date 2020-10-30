import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# GRADED FUNCTION: train_mnist
def train_mnist():
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('accuracy') > 1.00):
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 28, 28, 1)
    x_train= x_train / 255.0
    x_test = x_test.reshape(10000, 28, 28, 1)
    x_test = x_test / 255.0
    print(x_train.shape)
    print(x_test.shape)
    # plt.figure(figsize=(5, 5))
    # plt.imshow(x_train[1], cmap=plt.cm.binary)  # here cmap helps us to get the image in grey scale
    # plt.show()

    # plt.figure(figsize=(10, 10))
    # for i in range(25):
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(x_train[i], cmap=plt.cm.binary)
    #     plt.xlabel([y_train[i]])
    # plt.show()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # model fitting
    model.fit(x_train, y_train, epochs=5, callbacks=[callbacks])
    # model fitting
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("loss: ", test_loss)
    print("Acurracy: ", test_acc)
    return test_acc, test_loss



train_mnist()

