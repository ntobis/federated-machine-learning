# Imports
from __future__ import absolute_import, division, print_function, unicode_literals

import os

import matplotlib.pyplot as plt
import tensorflow as tf

models = tf.keras.models  # like 'from tensorflow.keras import models' (PyCharm import issue workaround)
layers = tf.keras.layers  # like 'from tensorflow.keras import layers' (PyCharm import issue workaround)

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------ Paths ----------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS = os.path.join(ROOT, 'Models')
CENTRALIZED_CHECK_POINT = os.path.join(MODELS, "Centralized Model", "centralized_cp.ckpt")
CENTRALIZED_MODEL = os.path.join(MODELS, "Centralized Model", "centralized_model.h5")

# ---------------------------------------------------- End Paths --------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Utility Functions ----------------------------------------------- #


def load_MNIST_data():
    """
    Loads the MNIST Data Set and reshapes it for further model training

    :param:

    :return:
        train_images    - numpy array of shape (60000, 28, 28, 1)
        train_labels    - numpy array of shape (60000, )
        test_images     - numpy array of shape (10000, 28, 28, 1)
        test_labels     - numpy array of shape (10000, )
    """

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    return train_images, train_labels, test_images, test_labels


def create_checkpoint_callback():
    """
    Creates a checkpoint callback at specified file location, to be passed into the model.fit() function

    :param:

    :return:
        cp_callback     - tensorflow callback function object
    """

    cp_callback = tf.keras.callbacks.ModelCheckpoint(CENTRALIZED_CHECK_POINT,
                                                     save_weights_only=True,
                                                     verbose=1)
    return cp_callback


# ---------------------------------------------- End Utility Functions --------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# ----------------------------------------------- Model Configuration ---------------------------------------------- #

def build_cnn(input_shape):
    """
    Compile and return a simple CNN model for image recognition.

    Configuration:
    Layer 1: Convolution Layer | Filters: 32 | Kernel Size: 3x3 | Activation: Relu
    Layer 2: Max Pooling Layer | Filter: 2x2
    Layer 3: Dense Layer       | Neurons: 32 | Activation: Relu
    Layer 4: Dense Layer       | Neurons: 10 | Activation: Softmax

    Optimizer:      Adam
    Loss function:  Sparse Categorical Cross Entropy
    Loss metric:    Accuracy


    :param input_shape:     image input shape (tuple), e.g. (28, 28, 1)

    :return:
        model               compiled tensorflow model
    """

    # Set up model type
    model = models.Sequential()

    # Add layers, inspired by https://www.tensorflow.org/beta/tutorials/images/intro_to_cnns
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train_cnn(model, train_data, train_labels, epochs=2, callbacks=None):
    """
    Train and return a simple CNN model for image recognition

    :param model:           compiled tensorflow model
    :param train_data:      numpy array
    :param train_labels:    numpy array
    :param epochs:          number of training epochs (i.e. iterations over train_data)
    :param callbacks:       array of callback functions

    :return:
         model              trained tensorflow model
    """

    model.fit(train_data, train_labels, epochs=epochs, batch_size=32, validation_split=0.25, callbacks=callbacks)
    return model


def evaluate_cnn(model, test_data, test_labels):
    """
    Evaluate and return a simple CNN model for image recognition

    :param model:           compiled tensorflow model
    :param test_data:       numpy array
    :param test_labels:     numpy array

    :return:
        test_loss           float
        test_acc            float (TP / All Observations)
    """

    test_loss, test_acc = model.evaluate(test_data, test_labels)
    return test_loss, test_acc


# --------------------------------------------- End Model Configuration -------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# ----------------------------------------------- Plotting and Display --------------------------------------------- #


def plot_accuracy(model):
    """
    Plot training & validation accuracy values over epochs

    :param model:           trained tensorflow model holding a 'History' objects

    :return:

    """

    try:  # Depending on the TF version, these are labeled differently
        plt.plot(model.history.history['accuracy'])  # 'accuracy'
        plt.plot(model.history.history['val_accuracy'])
    except KeyError:
        plt.plot(model.history.history['acc'])  # 'acc'
        plt.plot(model.history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def plot_loss(model):
    """
    Plot training & validation loss values over epochs

    :param model:           trained tensorflow model holding a 'History' objects

    :return:

    """

    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def display_images(train_data, train_labels):
    """
    Display first 9 MNIST images

    :param train_data:      numpy array of shape (60000, 28, 28, 1)
    :param train_labels:    numpy array of shape (60000, )

    :return:

    """

    train_data = tf.reshape(train_data, [60000, 28, 28])
    # Display Digits
    fig = plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(train_data[i], cmap='gray', interpolation='none')
        plt.title("Digit: {}".format(train_labels[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


# --------------------------------------------- End Plotting and Display ------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

def main(plotting=False, training=True, loading=False, evaluating=True, max_samples=None):
    """
    Main function including a number of flags that can be set

    :param plotting:            bool
    :param training:            bool
    :param loading:             bool
    :param evaluating:          bool
    :param max_samples:         int

    :return:

    """
    # Load data
    train_images, train_labels, test_images, test_labels = load_MNIST_data()

    if max_samples:
        train_images = train_images[:max_samples]
        train_labels = train_labels[:max_samples]

    # Display data
    if plotting:
        display_images(train_images, train_labels)

    # Enable check-pointing
    cp_callback = create_checkpoint_callback()

    # Build model
    model = build_cnn(input_shape=(28, 28, 1))
    if loading:
        model.load_weights(CENTRALIZED_CHECK_POINT)

    # Train model
    if training:
        model = train_cnn(model, train_images, train_labels, [cp_callback])
        # Save full model
        model.save(CENTRALIZED_MODEL)

    # Evaluate model
    if evaluating:
        test_loss, test_acc = evaluate_cnn(model, test_images, test_labels)
        print("Test Loss: {:5.2f}".format(test_loss))
        print("Test Accuracy: {:5.2f}%".format(100 * test_acc))

    # Plot Accuracy and Loss
    if plotting:
        plot_accuracy(model)
        plot_loss(model)


if __name__ == '__main__':
    main(training=True, loading=False, max_samples=None)
