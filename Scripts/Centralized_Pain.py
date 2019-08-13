# Imports
from __future__ import absolute_import, division, print_function, unicode_literals

import os

import numpy as np
import pandas as pd
import tensorflow as tf

from Scripts import Data_Loader_Functions as dL
from Scripts import Keras_Custom as kC

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
# ------------------------------------------------- Set-Up Functions ----------------------------------------------- #


def set_up_train_test_generators(df, model, session, balance_train):
    train_gen, predict_gen, df_train, df_test = [None] * 4  # Initializing values
    if df is not None:
        df_train = df[df['Session'] <= session]
        df_test = df[(df['Trans_1'] == 'original') & (df['Trans_2'] == 'straight')]
        if sum(df_train['Pain'] != '0'):
            train_gen = set_up_data_generator(df_train, model.name, shuffle=True, balanced=balance_train,
                                              gen_type='Train')
        if len(df_test):
            predict_gen = set_up_data_generator(df_test, model.name, shuffle=False, balanced=False, gen_type='Test')
    return df_train, df_test, train_gen, predict_gen


def set_up_data_generator(df, model_name, shuffle=True, balanced=False, gen_type='Data_Gen'):
    print("{}: Actual number of images: ".format(gen_type), len(df), "thereof pain: ", sum(df['Pain'] != '0'))

    # Balance data
    if balanced:
        df = dL.balance_data(df, threshold=200)
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    # Ensure that input channels are of correct size (1-channel for 'CNN', 3-channels for 'ResNet'
    color_mode = 'rgb' if model_name is 'ResNet' else 'grayscale'
    return data_gen.flow_from_dataframe(dataframe=df, x_col="img_path",
                                        y_col="Pain", color_mode=color_mode,
                                        class_mode="categorical", target_size=(215, 215),
                                        batch_size=32,
                                        classes=['0', '1'], shuffle=shuffle)


def set_up_history():
    history = pd.DataFrame(columns=['Epoch', 'Loss', 'Session', 'Person', 'TN', 'FP', 'FN', 'TP',
                                    'Individual Avg. Precision', 'Aggregate Avg. Precision',
                                    'Individual Accuracy', 'Individual Precision', 'Individual Recall',
                                    'Individual F1-Score', 'Aggregate Accuracy', 'Aggregate Precision',
                                    'Aggregate Recall', 'Aggregate F1_Score'])
    return history


# ----------------------------------------------- End Set-Up Functions --------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------------------------ #
# -------------------------------------------------- Model Runners ------------------------------------------------- #


def train_cnn(algorithm, model, epochs, train_data=None, train_labels=None, test_data=None, test_labels=None,
              people=None, all_labels=None):
    # Create callbacks
    history_cb = None
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto',
                                                      baseline=None, restore_best_weights=True)
    callbacks = [early_stopping]

    # Create validation sets
    if test_data is not None:
        _, test_data_split, test_labels_split = dL.split_data_into_labels(0, all_labels, False, test_data, test_labels)
        validation_sets = [(val_data, val_labels, 'subject_{}'.format(person)) for val_data, val_labels, person in
                           zip(test_data_split, test_labels_split, np.unique(people))]
        history_cb = kC.AdditionalValidationSets(validation_sets)
        callbacks.append(history_cb)

    # Train and evaluate
    validation_split, validation_data = None, None
    if test_data is None and algorithm == 'centralized':  # This applies to pre-training a centralized model
        validation_split = 0.2
    elif test_data is not None:  # This applies to training any model
        validation_data = (test_data, test_labels)
    model.fit(train_data, train_labels, epochs=epochs, batch_size=32, use_multiprocessing=True,
              validation_split=validation_split, validation_data=validation_data, callbacks=callbacks)

    return model, history_cb.history if history_cb is not None else {}

# def evaluate_pain_cnn(model, epoch, test_data=None, test_labels=None, predict_gen=None, history=None, people=None,
#                       loss=None, df_test=None):
#     print('Evaluating')
#     if history is None:
#         history = set_up_history()
#
#     if test_data is not None:
#         predictions = model.predict(test_data, use_multiprocessing=True)
#         current_loss = compute_loss(loss, predictions, test_labels)
#         y_pred = np.argmax(predictions, axis=1)
#         df_metrics = compute_metrics(current_loss, epoch, people, predictions, test_labels, y_pred)
#
#     elif df_test is not None:
#         # Instantiate OneHotEncoder and encode Pain Labels
#         enc = OneHotEncoder(sparse=False, categories=[range(2)])
#         test_labels = enc.fit_transform(df_test['Pain'].astype(int).values.reshape(len(df_test), 1))
#         predictions = model.predict_generator(predict_gen, use_multiprocessing=True)
#         print("Done Predicting.")
#         current_loss = compute_loss(loss, predictions, test_labels)
#         y_pred = np.argmax(predictions, axis=1)
#         df_metrics = compute_metrics(current_loss, epoch, df_test['Person'], predictions, test_labels,
#                                      y_pred, df_test['Session'])
#
#     else:
#         raise ValueError('Either "test_data" or "df_test" must be not None.')
#     print('Done Evaluating')
#     return pd.concat((history, df_metrics), ignore_index=True, sort=False)
#
#
# # ------------------------------------------------ End Model Runners ----------------------------------------------- #
# # ------------------------------------------------------------------------------------------------------------------ #
#
#
# # ------------------------------------------------------------------------------------------------------------------ #
# # ----------------------------------------------- Evaluation Functions --------------------------------------------- #
#
# def compute_metrics(current_loss, epoch, people, predictions, test_labels, y_pred, sessions=None):
#     df = compute_individual_metrics(epoch, current_loss, people, test_labels, y_pred, predictions, sessions)
#     df = compute_aggregate_metrics(df, test_labels, predictions)
#     return df
#
#
# def compute_loss(loss, predictions, test_labels):
#     return loss(
#         tf.convert_to_tensor(tf.cast(test_labels, tf.float32)),
#         tf.convert_to_tensor(tf.cast(predictions, tf.float32))
#     ).numpy()
#
#
# def compute_individual_metrics(epoch, loss, people, test_labels, y_pred, predictions, sessions=None):
#     test_labels = test_labels[:, 1]
#     predictions = predictions[:, 1]
#
#     sessions = [0] * len(y_pred) if sessions is None else sessions
#     data = np.concatenate([np.expand_dims(x, 1) for x in [people, y_pred, test_labels, predictions, sessions]],
#     axis=1)
#     df = pd.DataFrame(data, columns=['Person', 'Y_Pred', 'Y_True', 'Predictions', 'Session'])
#
#     results = []
#     for session, df_session in df.groupby('Session'):
#         for person, df_person in df_session.groupby('Person'):
#             tn, fp, fn, tp = confusion_matrix(df_person['Y_True'], df_person['Y_Pred'], labels=[0, 1]).ravel()
#             ind_avg_precision = average_precision_score(df_person['Y_True'], df_person['Predictions'])
#             results.append([epoch, loss, session, person, tn, fp, fn, tp, ind_avg_precision])
#     df = pd.DataFrame(results, columns=['Epoch', 'Loss', 'Session', 'Person', 'TN', 'FP', 'FN', 'TP',
#                                         'Individual Avg. Precision'])
#
#     df['Individual Accuracy'] = (df['TP'] + df['TN']) / (df['TN'] + df['FP'] + df['FN'] + df['TP'])
#     df['Individual Precision'] = df['TP'] / (df['FP'] + df['TP'])
#     df['Individual Recall'] = df['TP'] / (df['FN'] + df['TP'])
#     df['Individual F1-Score'] = 2 * ((df['Individual Precision'] * df['Individual Recall']) / (
#             df['Individual Precision'] + df['Individual Recall']))
#     return df
#
#
# def compute_aggregate_metrics(df, test_labels, predictions):
#     test_labels = test_labels[:, 1]
#     predictions = predictions[:, 1]
#
#     df['Aggregate Avg. Precision'] = average_precision_score(test_labels, predictions)
#     tn_g = df.groupby('Epoch')['TN'].transform(sum)
#     fp_g = df.groupby('Epoch')['FP'].transform(sum)
#     fn_g = df.groupby('Epoch')['FN'].transform(sum)
#     tp_g = df.groupby('Epoch')['TP'].transform(sum)
#     df['Aggregate Accuracy'] = (tp_g + tn_g) / (tn_g + fp_g + fn_g + tp_g)
#     df['Aggregate Precision'] = tp_g / (fp_g + tp_g)
#     df['Aggregate Recall'] = tp_g / (fn_g + tp_g)
#     df['Aggregate F1_Score'] = 2 * ((df['Aggregate Precision'] * df['Aggregate Recall']) / (
#             df['Aggregate Precision'] + df['Aggregate Recall']))
#     return df
#
# # --------------------------------------------- End Evaluation Functions ------------------------------------------- #
# # ------------------------------------------------------------------------------------------------------------------ #
