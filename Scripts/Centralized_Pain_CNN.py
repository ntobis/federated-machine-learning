# Imports
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from Scripts import Data_Loader_Functions as dL
from Scripts import Centralized_CNN as cNN

models = tf.keras.models  # like 'from tensorflow.keras import models' (PyCharm import issue workaround)
layers = tf.keras.layers  # like 'from tensorflow.keras import layers' (PyCharm import issue workaround)
optimizers = tf.keras.optimizers  # like 'from tensorflow.keras import optimizers' (PyCharm import issue workaround)

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------ Paths ----------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "Data")
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
    model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), input_shape=input_shape))
    # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2)))
    # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2)))
    # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=128, activation='relu'))
    model.add(layers.Dense(units=2, activation='softmax'))

    return model


def train_cnn(model, train_data, train_labels, test_data, test_labels, metrics, epochs):
    results = []
    for epoch in range(epochs):

        # Training
        model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_data, train_labels, epochs=1, batch_size=32, use_multiprocessing=True)

        # Evaluating (Batch Size must be 1, otherwise TF throws an error)
        model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=metrics)
        epoch_results = model.evaluate(test_data, test_labels, batch_size=1)
        results.append(epoch_results)

        # Create DF for Progress
        df = pd.DataFrame(results, columns=['Loss',
                                            'Accuracy',
                                            'Recall',
                                            'Precision',
                                            'AUC',
                                            'TP',
                                            'TN',
                                            'FP',
                                            'FN'
                                            ]
                          )
        f1_score = 2 * ((df['Precision'] * df['Recall']) / (df['Precision'] + df['Recall']))
        df['F1_Score'] = f1_score

        # Save Progress
        file = time.strftime("%Y-%m-%d-%H%M%S") + r'log_results_epoch-{}.csv'.format(epoch)
        df.to_csv(os.path.join(cNN.RESULTS, file))


def main(flag="TRAIN"):
    tf.random.set_seed(123)

    # Define paths
    group_1 = os.path.join(DATA, "Augmented Data", "Pain Two-Step Augmentation", "group_1")
    group_2 = os.path.join(DATA, "Augmented Data", "Pain Two-Step Augmentation", "group_2")
    test = os.path.join(DATA, "Augmented Data", "Pain Two-Step Augmentation", "test")

    # Define labels for training
    label = 4  # Labels: [person, session, culture, frame, pain, augmented]

    # Define Metrics
    metrics = [
        tf.metrics.Accuracy(),
        # tf.metrics.Recall(),
        # tf.metrics.Precision(),
        # tf.metrics.AUC(curve='PR'),
        # tf.metrics.TruePositives(),
        # tf.metrics.TrueNegatives(),
        # tf.metrics.FalsePositives(),
        # tf.metrics.FalseNegatives(),
    ]

    # 1. Cold start: Don't use train group
    # 2. Warm start: Use train group
    # 3. Level up from 0 % - 60%

    if flag == 'TRAIN':
        # # Load data
        # print("Loading Data")
        # data_1, labels_1, data_2, labels_2 = dL.load_pain_data(group_1, group_2)
        #
        # # Concatenate Data
        # data = np.concatenate((data_1, data_2))
        # labels = np.concatenate((labels_1, labels_2))
        #
        # # Reassign labels
        # labels_ord = labels[:, label].astype(np.int)
        # labels_bin = dL.reduce_pain_label_categories(labels_ord, max_pain=1)

        # Load data
        print("Loading Data")
        data, labels = dL.load_pain_data(group_1)

        # Reassign labels
        labels_ord = labels[:, label].astype(np.int)
        labels_bin = dL.reduce_pain_label_categories(labels_ord, max_pain=1)

        # Initialize model
        print("Initializing model")
        pain_model = build_cnn(data[0].shape)

        # Compile model
        pain_model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=metrics)

        # Define Callbacks
        check_points = tf.keras.callbacks.ModelCheckpoint(os.path.join(PAIN_MODELS,
                                                                       "weights.{epoch:02d}-{val_loss:.2f}.hdf5"))
        # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        # Fit model
        print("Fitting model")
        pain_model.fit(data, labels_bin, epochs=30, batch_size=32, validation_split=0.3,
                       use_multiprocessing=True, callbacks=[check_points])

    if flag == 'TEST':
        # Load and evaluate model
        test_data, test_labels = dL.load_pain_data(test)

        # Reassign labels
        test_labels_ord = test_labels[:, label].astype(np.int)
        test_labels_bin = dL.reduce_pain_label_categories(test_labels_ord, max_pain=1)

        pain_model = build_cnn(test_data[0].shape)
        pain_model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        pain_model.load_weights(
            "/Users/nico/PycharmProjects/FederatedLearning/Models/Centralized Pain/weights.30-0.34.hdf5")

        pain_model.evaluate(test_data, test_labels_bin)
