# Imports
from __future__ import absolute_import, division, print_function, unicode_literals

import os

import numpy as np
import tensorflow as tf

from Scripts import Print_Functions as Output
from Scripts import Data_Loader_Functions as Data_Loader

models = tf.keras.models  # like 'from tensorflow.keras import models' (PyCharm import issue workaround)
layers = tf.keras.layers  # like 'from tensorflow.keras import layers' (PyCharm import issue workaround)
optimizers = tf.keras.optimizers  # like 'from tensorflow.keras import optimizers' (PyCharm import issue workaround)

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------ Paths ----------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS = os.path.join(ROOT, 'Models')
AUTISM = os.path.join(ROOT, 'Data', 'Autism')
CENTRALIZED_MODEL_PATH = os.path.join(MODELS, "Centralized Model")
CENTRALIZED_CHECK_POINT = os.path.join(CENTRALIZED_MODEL_PATH, "centralized_cp.ckpt")
CENTRALIZED_MODEL = os.path.join(CENTRALIZED_MODEL_PATH, "centralized_model.h5")
RESULTS = os.path.join(ROOT, 'Results')


# ---------------------------------------------------- End Paths --------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------------------------ #
# ----------------------------------------------- Model Configuration ---------------------------------------------- #

def create_checkpoint_callback():
    """
    Creates a checkpoint callback at specified file location, to be passed into the model.fit() function

    :return:
        cp_callback     - tensorflow callback function object
    """

    cp_callback = tf.keras.callbacks.ModelCheckpoint(CENTRALIZED_CHECK_POINT,
                                                     save_weights_only=True,
                                                     verbose=1)
    return cp_callback


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
    model.add(layers.Conv2D(32, (5, 5), input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5), input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model


def train_cnn(model, train_data, train_labels, epochs=30, callbacks=None):
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

    model.fit(train_data, train_labels, epochs=epochs, batch_size=10, validation_split=0, callbacks=callbacks)
    return model


# --------------------------------------------- End Model Configuration -------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


def evaluate_cnn(model, test_data, test_labels, train_data=None, train_labels=None):
    """
    Evaluate and return a simple CNN model for image recognition

    :param model:           compiled tensorflow model
    :param test_data:       numpy array
    :param test_labels:     numpy array
    :param train_labels:    numpy array, optional
    :param train_data:      numpy array, optional

    :return:
        test_loss           float
        test_acc            float (TP / All Observations)
    """
    test_loss, test_acc = model.evaluate(test_data, test_labels)

    if train_data is not None and train_labels is not None:
        train_loss, train_acc = model.evaluate(np.concatenate(train_data), np.concatenate(train_labels))
        return test_loss, test_acc, train_loss, train_acc

    return test_loss, test_acc


def main(plotting=False, training=True, loading=False, evaluating=True, max_samples=None, dataset='MNIST'):
    """
    Main function including a number of flags that can be set

    :param dataset:             string, specifying the dataset to be used
    :param plotting:            bool
    :param training:            bool
    :param loading:             bool
    :param evaluating:          bool
    :param max_samples:         int

    """
    train_data, train_labels, test_data, test_labels, dataset = Data_Loader.load_data(dataset)

    if max_samples:
        train_data = train_data[:max_samples]
        train_labels = train_labels[:max_samples]

    # Display data
    # if plotting:
    #     Output.display_images(train_data, train_labels)

    # Enable check-pointing
    cp_callback = create_checkpoint_callback()

    # Build model
    model = build_cnn(input_shape=(28, 28, 1))
    # Compile the model
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    if loading:
        model.load_weights(CENTRALIZED_CHECK_POINT)

    # Train model
    if training:
        model = train_cnn(model, train_data, train_labels, callbacks=[cp_callback])
        # Save full model
        model.save(CENTRALIZED_MODEL)

    # Evaluate model
    if evaluating:
        test_loss, test_acc = evaluate_cnn(model, test_data, test_labels)
        print("Test Loss: {:5.2f}".format(test_loss))
        print("Test Accuracy: {:5.2%}".format(test_acc))

    # Plot Accuracy and Loss
    if plotting:
        Output.plot_centralized_accuracy(model)
        Output.plot_centralized_loss(model)


if __name__ == '__main__':
    main(training=True, plotting=True, loading=False, max_samples=None)
