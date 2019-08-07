# Imports
from __future__ import absolute_import, division, print_function, unicode_literals

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, average_precision_score

from Scripts import Centralized_CNN as cNN
from Scripts import Data_Loader_Functions as dL

models = tf.keras.models  # like 'from tensorflow.keras import models' (PyCharm import issue workaround)
layers = tf.keras.layers  # like 'from tensorflow.keras import layers' (PyCharm import issue workaround)
optimizers = tf.keras.optimizers  # like 'from tensorflow.keras import optimizers' (PyCharm import issue workaround)

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------ Paths ----------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(__file__))
SESSION_DATA = os.path.join(ROOT, "Data", "Augmented Data", "Flexible Augmentation", "group_2")
EPOCH_LEN = 1

# ---------------------------------------------------- End Paths --------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------------------------ #
# ----------------------------------------------- Model Configuration ---------------------------------------------- #


class EarlyStopping:
    def __init__(self, metric, threshold, patience):
        self.metric = metric
        self.threshold = threshold
        self.patience = patience

    def __call__(self, history):
        if self.metric == 'accuracy':
            return len(history) >= self.patience and all(x > self.threshold for x in history[-self.patience:])
        else:
            return False


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
    model.add(layers.Dense(units=128))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dense(units=2, activation='sigmoid'))

    return model


def train_cnn(model, epochs, train_data=None, train_labels=None, test_data=None, test_labels=None, df=None,
              people=None, evaluate=True, loss=None, early_stopping=None, session=None):
    # Set up data frames for logging
    history = history_set_up(people)
    early_stopping_hist = []

    # Start training
    for epoch in range(epochs):

        # Training (either on dataset, or on Keras train iterator
        if train_data is not None and train_labels is not None:
            train_hist = model.fit(train_data, train_labels, epochs=1, batch_size=32, use_multiprocessing=True)
        elif df is not None and session is not None:
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
            train_df = df[df['Session'] <= session]
            print("Actual number of images: ", len(train_df), "thereof pain: ", sum(train_df['Pain'] != '0'))
            train_df = dL.balance_data(train_df, threshold=200)

            # Only train this client, if pain images exist
            if len(train_df) > 0:
                train_gen = data_gen.flow_from_dataframe(dataframe=train_df, directory=SESSION_DATA, x_col="img_path",
                                                         y_col="Pain", color_mode="grayscale",
                                                         class_mode="categorical", target_size=(215, 215), batch_size=32,
                                                         classes=['0', '1'])
                train_hist = model.fit_generator(generator=train_gen, steps_per_epoch=train_gen.n // train_gen.batch_size,
                                                 epochs=1)
        else:
            raise KeyError("Need to specify either ('train_data' and 'train_labels') or 'df'. Neither was specified.")

        # Evaluating
        if evaluate:
            print('Evaluating')
            history = evaluate_pain_cnn(model, epoch, test_data, test_labels, df, history, people, loss)

        # Early stopping
        if early_stopping is not None:
            early_stopping_hist.extend(train_hist.history[early_stopping.metric])
            if early_stopping(train_hist):
                break

    return model, history


def history_set_up(people):
    if people is not None:
        history = pd.DataFrame(columns=['Epoch', 'Loss', 'Session', 'Person', 'TN', 'FP', 'FN', 'TP',
                                        'Individual Avg. Precision', 'Aggregate Avg. Precision',
                                        'Individual Accuracy', 'Individual Precision', 'Individual Recall',
                                        'Individual F1-Score', 'Aggregate Accuracy', 'Aggregate Precision',
                                        'Aggregate Recall', 'Aggregate F1_Score'])
    else:
        history = pd.DataFrame(columns=['Epoch', 'Loss', 'Session', 'Aggregate Accuracy', 'Aggregate Recall',
                                        'Aggregate Precision', 'Aggregate Avg. Precision', 'TP', 'TN', 'FP',
                                        'FN', 'Aggregate F1_Score'])
    return history


def evaluate_pain_cnn(model, epoch, test_data=None, test_labels=None, df=None, history=None, people=None, loss=None):
    if history is None:
        history = history_set_up(people)

    if df is not None:
        all_predictions = []
        all_labels = []
        df_test = df[(df['Trans_1'] == 'original') & (df['Trans_2'] == 'straight')]
        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        for sess, df_sess in df_test.groupby('Session'):
            print("\nEvaluate Session :", sess)
            for person, df_person in df_sess.groupby('Person'):
                predict_gen = data_gen.flow_from_dataframe(dataframe=df_person, directory=SESSION_DATA,
                                                           x_col="img_path",
                                                           y_col="Pain", color_mode="grayscale",
                                                           class_mode="categorical", target_size=(215, 215),
                                                           batch_size=32, classes=['0', '1'])

                # Get predictions and labels for specific person/session combination
                predictions = []
                labels = []
                # EPOCH_LEN = predict_gen.n // predict_gen.batch_size
                for i in range(predict_gen.n // predict_gen.batch_size):
                    x, y = next(predict_gen)
                    predictions.append(model.predict(x))
                    labels.append(y)
                predictions = np.concatenate(predictions)
                labels = np.concatenate(labels)

                # Get Loss
                current_loss = loss(
                    tf.convert_to_tensor(tf.cast(labels, tf.float32)),
                    tf.convert_to_tensor(tf.cast(predictions, tf.float32))
                ).numpy()

                # Get y_pred
                y_pred = np.argmax(predictions, axis=1)

                # Compute individual metrics
                df = compute_individual_metrics(epoch, current_loss, person, labels, y_pred, predictions, sess)
                history = history.append(df, ignore_index=True)
                all_predictions.append(predictions)
                all_labels.append(labels)

        # Compute aggregate metrics
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        history = compute_aggregate_metrics(history, all_labels, all_predictions)
    else:
        predictions = model.predict(test_data)
        current_loss = loss(
            tf.convert_to_tensor(tf.cast(test_labels, tf.float32)),
            tf.convert_to_tensor(tf.cast(predictions, tf.float32))
        ).numpy()
        y_pred = np.argmax(predictions, axis=1)

        # If people were passed, compute metrics on a per person basis as well as aggregate
        # Else just compute aggregate
        df = compute_individual_metrics(epoch, current_loss, people, test_labels, y_pred, predictions)
        history = history.append(df, ignore_index=True)

    # Save logs
    if people is not None:
        file = r'logs_individual.csv'
        history.to_csv(os.path.join(cNN.RESULTS, file))
    else:
        file = r'logs_aggregate.csv'
        history.to_csv(os.path.join(cNN.RESULTS, file))
    return history


def compute_individual_metrics(epoch, loss, people, test_labels, y_pred, predictions, session=None):
    test_labels = test_labels[:, 1]
    predictions = predictions[:, 1]

    if type(people) is not int:
        data = np.concatenate([np.expand_dims(x, 1) for x in [people, y_pred, test_labels, predictions]], axis=1)
        df = pd.DataFrame(data, columns=['Person', 'Y_Pred', 'Y_True', 'Predictions'])
    else:
        data = np.concatenate([np.expand_dims(x, 1) for x in [y_pred, test_labels, predictions]], axis=1)
        df = pd.DataFrame(data, columns=['Y_Pred', 'Y_True', 'Predictions'])
        df['Person'] = people

    results = []
    for person in df['Person'].unique():
        df_person = df[df['Person'] == person]
        tn, fp, fn, tp = confusion_matrix(df_person['Y_True'], df_person['Y_Pred'], labels=[0, 1]).ravel()
        ind_avg_precision = average_precision_score(df_person['Y_True'], df_person['Predictions'])
        results.append([epoch, loss, session, person, tn, fp, fn, tp, ind_avg_precision])
    df = pd.DataFrame(results, columns=['Epoch', 'Loss', 'Session', 'Person', 'TN', 'FP', 'FN', 'TP',
                                        'Individual Avg. Precision'])

    df['Individual Accuracy'] = (df['TP'] + df['TN']) / (df['TN'] + df['FP'] + df['FN'] + df['TP'])
    df['Individual Precision'] = df['TP'] / (df['FP'] + df['TP'])
    df['Individual Recall'] = df['TP'] / (df['FN'] + df['TP'])
    df['Individual F1-Score'] = 2 * ((df['Individual Precision'] * df['Individual Recall']) / (
            df['Individual Precision'] + df['Individual Recall']))
    return df


def compute_aggregate_metrics(df, test_labels, predictions):
    test_labels = test_labels[:, 1]
    predictions = predictions[:, 1]

    df['Aggregate Avg. Precision'] = average_precision_score(test_labels, predictions)
    tn_g = df.groupby('Epoch')['TN'].transform(sum)
    fp_g = df.groupby('Epoch')['FP'].transform(sum)
    fn_g = df.groupby('Epoch')['FN'].transform(sum)
    tp_g = df.groupby('Epoch')['TP'].transform(sum)
    df['Aggregate Accuracy'] = (tp_g + tn_g) / (tn_g + fp_g + fn_g + tp_g)
    df['Aggregate Precision'] = tp_g / (fp_g + tp_g)
    df['Aggregate Recall'] = tp_g / (fn_g + tp_g)
    df['Aggregate F1_Score'] = 2 * ((df['Aggregate Precision'] * df['Aggregate Recall']) / (
            df['Aggregate Precision'] + df['Aggregate Recall']))
    return df
