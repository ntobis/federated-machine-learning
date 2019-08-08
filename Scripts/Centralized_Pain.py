# Imports
from __future__ import absolute_import, division, print_function, unicode_literals

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, average_precision_score

from Scripts import Data_Loader_Functions as dL

models = tf.keras.models  # like 'from tensorflow.keras import models' (PyCharm import issue workaround)
layers = tf.keras.layers  # like 'from tensorflow.keras import layers' (PyCharm import issue workaround)
optimizers = tf.keras.optimizers  # like 'from tensorflow.keras import optimizers' (PyCharm import issue workaround)

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------ Paths ----------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(__file__))
SESSION_DATA = os.path.join(ROOT, "Data", "Augmented Data", "Flexible Augmentation", "group_2")
RESULTS = os.path.join(ROOT, 'Results')


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


def train_cnn(model, epochs, train_data=None, train_labels=None, test_data=None, test_labels=None,
              df=pd.DataFrame(columns=['Session']), people=None, evaluate=True, loss=None, session=-1):
    # Set up data frames for logging
    history = history_set_up(people)

    # Set up Keras data generators
    df_train = df[df['Session'] <= session]
    train_gen = set_up_data_generator(df_train, model.name) if len(df_train) > 0 else None

    df_test, predict_gen = set_up_predict_generator(df, evaluate, model)

    # Start training
    for epoch in range(epochs):

        # Training (on dataset)
        if train_data is not None and train_labels is not None:
            model.fit(train_data, train_labels, epochs=1, batch_size=32, use_multiprocessing=True)

        # Training (on Keras iterator)
        elif train_gen is not None:
            model.fit_generator(generator=train_gen,
                                steps_per_epoch=train_gen.n // train_gen.batch_size,
                                epochs=1, use_multiprocessing=True)
        else:
            print("Not training, since input data was empty.")

        # Evaluating
        if evaluate:
            print('Evaluating')
            history = evaluate_pain_cnn(model, epoch, test_data, test_labels, predict_gen, history, people, loss,
                                        df_test)

    return model, history


def set_up_predict_generator(df, evaluate, model):
    df_test = df[(df['Trans_1'] == 'original') & (df['Trans_2'] == 'straight')] if evaluate and len(df) > 0 else None
    predict_gen = set_up_data_generator(df_test, model.name, shuffle=False) if df_test is not None else None
    return df_test, predict_gen


def set_up_data_generator(df, model_name, shuffle=True):
    print("Actual number of images: ", len(df), "thereof pain: ", sum(df['Pain'] != '0'))

    # Balance data
    df = dL.balance_data(df, threshold=200)
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    # Ensure that input channels are of correct size (1-channel for 'CNN', 3-channels for 'ResNet'
    color_mode = 'rgb' if model_name is 'ResNet' else 'grayscale'

    return data_gen.flow_from_dataframe(dataframe=df, directory=SESSION_DATA, x_col="img_path",
                                        y_col="Pain", color_mode=color_mode,
                                        class_mode="categorical", target_size=(215, 215),
                                        batch_size=32,
                                        classes=['0', '1'], shuffle=shuffle)


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


def evaluate_pain_cnn(model, epoch, test_data=None, test_labels=None, predict_gen=None, history=None, people=None, loss=None, df_test=None):
    if history is None:
        history = history_set_up(people)

    if test_data is not None:
        predictions = model.predict(test_data, use_multiprocessing=True)
        current_loss = compute_loss(loss, predictions, test_labels)
        y_pred = np.argmax(predictions, axis=1)
        df_metrics = compute_metrics(current_loss, epoch, people, predictions, test_labels, y_pred)
        history = history.append(df_metrics, ignore_index=True)

    elif df_test is not None:
        predictions = model.predict_generator(predict_gen, use_multiprocessing=True)
        current_loss = compute_loss(loss, predictions, df_test['Pain'])
        y_pred = np.argmax(predictions, axis=1)
        df_metrics = compute_metrics(current_loss, epoch, df_test['Person'], predictions, df_test['Pain'].astype(int),
                                     y_pred)
        history = history.append(df_metrics, ignore_index=True)

    else:
        raise ValueError('Either "test_data" or "df_test" must be not None.')

    # Save logs
    if people is not None:
        file = r'logs_individual.csv'
        history.to_csv(os.path.join(RESULTS, file))
    else:
        file = r'logs_aggregate.csv'
        history.to_csv(os.path.join(RESULTS, file))
    return history


def compute_metrics(current_loss, epoch, people, predictions, test_labels, y_pred):
    df = compute_individual_metrics(epoch, current_loss, people, test_labels, y_pred, predictions)
    df = compute_aggregate_metrics(df, test_labels, predictions)
    return df


def compute_loss(loss, predictions, test_labels):
    return loss(
        tf.convert_to_tensor(tf.cast(test_labels, tf.float32)),
        tf.convert_to_tensor(tf.cast(predictions, tf.float32))
    ).numpy()


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
