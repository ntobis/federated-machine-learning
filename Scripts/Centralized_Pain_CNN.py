# Imports
from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
import numpy as np

from Scripts import Data_Loader_Functions as dL

models = tf.keras.models  # like 'from tensorflow.keras import models' (PyCharm import issue workaround)
layers = tf.keras.layers  # like 'from tensorflow.keras import layers' (PyCharm import issue workaround)
optimizers = tf.keras.optimizers  # like 'from tensorflow.keras import optimizers' (PyCharm import issue workaround)

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------ Paths ----------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "Data")
PAIN_TRAIN = os.path.join(DATA, "Augmented Data", "Pain", "training")
PAIN_TEST = os.path.join(DATA, "Augmented Data", "Pain", "test")
MODELS = os.path.join(ROOT, "Models")
PAIN_MODELS = os.path.join(MODELS, "Centralized Pain")


# ---------------------------------------------------- End Paths --------------------------------------------------- #
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

    # Add layers
    model.add(layers.Conv2D(32, (5, 5), input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (5, 5)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    return model


if __name__ == '__main__':
    # # Define labels for training
    # label = -2  # Labels: [person, session, culture, frame, pain, augmented]
    # max_pain_level = 1
    #
    # # Load data
    # print("Loading Data")
    # train_data, train_labels, test_data, test_labels = dL.load_pain_data(PAIN_TRAIN, '')
    #
    # # Reassign labels
    # train_labels = dL.reduce_pain_label_categories(train_labels[:, label].astype(np.int), max_pain=max_pain_level)
    # # test_labels = dL.reduce_pain_label_categories(test_labels[:, label].astype(np.int), max_pain=max_pain_level)
    #
    # Initialize model
    print("Initializing model")
    pain_model = build_cnn((250, 250, 1))

    # # Compile model
    # pain_model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #
    # # Define Callbacks
    # check_points = tf.keras.callbacks.ModelCheckpoint(os.path.join(PAIN_MODELS,
    #                                                                "weights.{epoch:02d}-{val_loss:.2f}.hdf5"))
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    #
    # # Fit model
    # print("Fitting model")
    # pain_model.fit(train_data[:500], train_labels[:500], epochs=30, batch_size=32, validation_split=0.1,
    #                use_multiprocessing=True, callbacks=[check_points, early_stopping])

    pain_model.load_weights('/Users/nico/PycharmProjects/FederatedLearning/Models/Centralized Painweights.08-0.57.hdf5')
