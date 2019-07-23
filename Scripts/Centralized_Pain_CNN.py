# Imports
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, average_precision_score

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
    model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2)))
    model.add(layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=128, activation='relu'))
    model.add(layers.Dense(units=2, activation='softmax'))

    return model


def train_cnn(model, train_data, train_labels, test_data, test_labels, epochs, people=None):

    # Set up data frames for logging
    df_individual = pd.DataFrame(columns=['Epoch', 'Loss', 'Person', 'TN', 'FP', 'FN', 'TP',
                                          'Individual Avg. Precision', 'Aggregate Avg. Precision',
                                          'Individual Accuracy', 'Individual Precision', 'Individual Recall',
                                          'Individual F1-Score', 'Aggregate Accuracy', 'Aggregate Precision',
                                          'Aggregate Recall', 'Aggregate F1-Score'])

    df_aggregate = pd.DataFrame(columns=['Epoch', 'Loss', 'Aggregate Accuracy', 'Aggregate Recall',
                                         'Aggregate Precision', 'Aggregate Avg. Precision', 'TP', 'TN', 'FP',
                                         'FN', 'Aggregate F1_Score'])

    # Start training
    for epoch in range(epochs):

        # Training
        model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_data, train_labels, epochs=1, batch_size=32, use_multiprocessing=True)

        # Evaluating (get predictions and loss)
        model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        predictions = model.predict(test_data)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        loss = loss(
            tf.convert_to_tensor(tf.cast(test_labels, tf.float32)),
            tf.convert_to_tensor(tf.cast(predictions, tf.float32))
        ).numpy()
        y_pred = np.argmax(predictions, axis=1)

        # If people were passed, compute metrics on a per person basis as well as aggregate
        # Else just compute aggregate
        if people is not None:
            df = compute_individual_metrics(epoch, loss, people, test_labels, y_pred, predictions)
            df_individual.append(df)

        else:
            df = compute_aggregate_metrics(epoch, loss, test_labels, y_pred, predictions)
            df_aggregate.append(df)

        # Save logs
        if people is not None:
            file = r'logs_individual.csv'
            df_individual.to_csv(os.path.join(cNN.RESULTS, file))
        else:
            file = r'logs_aggregate.csv'
            df_aggregate.to_csv(os.path.join(cNN.RESULTS, file))

    # Save Final Results
    if people is not None:
        file = time.strftime("%Y-%m-%d-%H%M%S") + r'_final_results_individual.csv'
        df_individual.to_csv(os.path.join(cNN.RESULTS, file))
    else:
        file = time.strftime("%Y-%m-%d-%H%M%S") + r'_final_results_aggregate.csv'
        df_individual.to_csv(os.path.join(cNN.RESULTS, file))
    return model


def compute_individual_metrics(epoch, loss, people, test_labels, y_pred, predictions):
    predictions = np.max(predictions, axis=1)
    data = np.concatenate([np.expand_dims(x, 1) for x in [people, y_pred, test_labels, predictions]], axis=1)
    df = pd.DataFrame(data, columns=['Person', 'Y_Pred', 'Y_True', 'Predictions'])

    results = []
    for person in df['Person'].unique():
        df_person = df[df['Person'] == person]
        tn, fp, fn, tp = confusion_matrix(df_person['Y_True'], df_person['Y_Pred']).ravel()
        ind_avg_precision = average_precision_score(df_person['Y_True'], df_person['Predictions'])
        aggregate_avg_precision = average_precision_score(df['Y_True'], df['Predictions'])
        results.append([epoch, loss, person, tn, fp, fn, tp, ind_avg_precision, aggregate_avg_precision])
    df = pd.DataFrame(results, columns=['Epoch', 'Loss', 'Person', 'TN', 'FP', 'FN', 'TP', 'Individual Avg. Precision',
                                        'Aggregate Avg. Precision'])

    df['Individual Accuracy'] = df['TP'] / (df['TN'] + df['FP'] + df['FN'] + df['TP'])
    df['Individual Precision'] = df['TP'] / (df['FP'] + df['TP'])
    df['Individual Recall'] = df['TP'] / (df['FN'] + df['TP'])
    df['Individual F1-Score'] = 2 * ((df['Individual Precision'] * df['Individual Recall']) / (
            df['Individual Precision'] + df['Individual Recall']))
    tn_g = df.groupby('Epoch')['TN'].transform(sum)
    fp_g = df.groupby('Epoch')['FP'].transform(sum)
    fn_g = df.groupby('Epoch')['FN'].transform(sum)
    tp_g = df.groupby('Epoch')['TP'].transform(sum)
    df['Aggregate Accuracy'] = tp_g / (tn_g + fp_g + fn_g + tp_g)
    df['Aggregate Precision'] = tp_g / (fp_g + tp_g)
    df['Aggregate Recall'] = tp_g / (fn_g + tp_g)
    df['Aggregate F1_Score'] = 2 * ((df['Aggregate Precision'] * df['Aggregate Recall']) / (
            df['Aggregate Precision'] + df['Aggregate Recall']))
    return df


def compute_aggregate_metrics(epoch, loss, test_labels, y_pred, predictions):
    predictions = np.max(predictions, axis=1)

    # Getting relevant metrics
    accuracy = accuracy_score(test_labels, y_pred)
    recall = recall_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred)
    tn, fp, fn, tp = confusion_matrix(test_labels, y_pred).ravel()
    aggregate_avg_precision = average_precision_score(test_labels, predictions)
    results = [epoch, loss, accuracy, recall, precision, aggregate_avg_precision, tp, tn, fp, fn]
    print(results)

    # Create DF for Progress
    df = pd.DataFrame([results], columns=['Epoch', 'Loss', 'Aggregate Accuracy', 'Aggregate Recall',
                                          'Aggregate Precision', 'Aggregate Avg. Precision', 'TP', 'TN', 'FP', 'FN'])
    df['Aggregate F1_Score'] = 2 * ((df['Aggregate Precision'] * df['Aggregate Recall']) / (df['Aggregate Precision'] +
                                                                                            df['Aggregate Recall']))
    return df
