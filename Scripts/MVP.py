import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tensorflow as tf
import numpy as np
from Scripts import Model_Architectures as mA
from Scripts import Experiments
from Scripts import Data_Loader_Functions as dL
import keras.backend as K

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "Data", "Augmented Data", "Flexible Augmentation")
GROUP_2_PATH = os.path.join(DATA, "group_2")
RESULTS = os.path.join(ROOT, "Results")


def weighted_loss(y_true, y_pred):
    weights = 1 / 0.2
    return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, weights)


def custom_metric(y_true, y_pred):
    return K.argmax(y_true, axis=1)


def TP(y_true, y_pred):
    y_pred = tf.argmax(y_pred, 1)
    y_true = tf.argmax(y_true, 1)
    return tf.math.count_nonzero(y_pred * y_true)


def FP(y_true, y_pred):
    y_pred = tf.argmax(y_pred, 1)
    y_true = tf.argmax(y_true, 1)
    return tf.math.count_nonzero(y_pred * (y_true - 1))


def TN(y_true, y_pred):
    y_pred = tf.argmax(y_pred, 1)
    y_true = tf.argmax(y_true, 1)
    return tf.math.count_nonzero((y_pred - 1) * (y_true - 1))


def FN(y_true, y_pred):
    y_pred = tf.argmax(y_pred, 1)
    y_true = tf.argmax(y_true, 1)
    return tf.math.count_nonzero((y_pred - 1) * y_true)


def conf_matrix(y_true, y_pred):
    return TP(y_true, y_pred), FP(y_true, y_pred), TN(y_true, y_pred), FN(y_true, y_pred)


def epoch_acc(y_true, y_pred):
    tp, fp, tn, fn = conf_matrix(y_true, y_pred)
    return (tp + tn) / (tp + fp + tn + fn)


def recall(y_true, y_pred):
    tp, fp, tn, fn = conf_matrix(y_true, y_pred)
    return tp / (fn + tp)


def precision(y_true, y_pred):
    tp, fp, tn, fn = conf_matrix(y_true, y_pred)
    return tp / (fp + tp)


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r)


def prec(y_true, y_pred):
    # y_true = tf.cast(tf.argmax(y_true, 1), tf.int64)
    # y_pred = tf.cast(y_pred, tf.int64)
    # y_true = tf.cast(y_true, tf.int64)
    return tf.compat.v1.metrics.precision_at_k(y_true, y_pred, 1)


def main():
    twilio = Experiments.Twilio()

    # Cumulative Training - DONE
    # Non-cumulative training - DONE
    # Training with just 1 person - DONE
    # Training with weighted loss
    # Training with moving training window for balancing non-cumulative
    # Training with moving training window for balancing cumulative
    # Repeat training with ResNet
    model_type = 'CNN'
    model = mA.build_model((215, 215, 1), model_type=model_type)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

    model.compile(optimizer, 'binary_crossentropy', ['accuracy', TP, TN, FP, FN])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto',
                                                      baseline=None, restore_best_weights=True)

    # for person in [43,  48,  52,  59,  64,  80,  92,  96, 107, 109, 115, 120]:
    train_data, train_labels_binary = None, None
    d = None
    train_df = dL.create_pain_df(GROUP_2_PATH)
    for idx, folder in enumerate(sorted(os.listdir(GROUP_2_PATH))):
        print("Session: {}".format(idx))
        f_path = os.path.join(GROUP_2_PATH, folder)
        df = dL.create_pain_df(f_path)
        # df = df[df['Person'] == person]
        f_paths = df['img_path'].values
        if len(f_paths) > 0:
            val_data, val_labels_binary, val_labels_people, val_labels = Experiments.load_and_prepare_data(f_paths, 0, 4,
                                                                                                           model_type)
            if idx > 0:
                print("Val_Balance: {:,.0%}".format(np.sum(val_labels_binary[:, 1]) / len(val_labels_binary)))
                history = model.fit(train_data, train_labels_binary, batch_size=32, epochs=30,
                                    validation_data=(val_data, val_labels_binary), callbacks=[early_stopping])
                if d is None:
                    d = pd.DataFrame(history.history)
                    d['Session'] = idx
                else:
                    d_new = pd.DataFrame(history.history)
                    d_new['Session'] = idx
                    d = pd.concat((d, d_new))

            df = train_df[train_df['Session'] <= idx]
            print("{}: Actual number of images: ".format(folder), len(df), "thereof pain: ", sum(df['Pain'] != '0'))
            df = dL.balance_data(df, threshold=200)
            train_data, train_labels_binary, train_labels_people, train_labels = Experiments.load_and_prepare_data(
                df['img_path'].values, 0, 4, model_type)
            # if idx <= 0:
            #     train_data, train_labels_binary = val_data, val_labels_binary
            # else:
            #     train_data, train_labels_binary = np.concatenate((train_data, val_data)), np.concatenate(
            #         (train_labels_binary, val_labels_binary))

    # file = os.path.join(RESULTS, 'ResNet Individual Training No Balancing Person {}.csv'.format(person))
    file = os.path.join(RESULTS, 'CNN Data Balancing.csv')
    d.to_csv(file)

    twilio.send_message("Training Done")


if __name__ == '__main__':
    main()
