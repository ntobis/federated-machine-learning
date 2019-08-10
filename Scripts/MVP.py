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
from sklearn.metrics import confusion_matrix, average_precision_score

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
    return K.sum(K.equal(y_pred, y_true))


def FP(y_true, y_pred):
    # y_pred = tf.argmax(y_pred, 1)
    # y_true = tf.argmax(y_true, 1)
    return tf.math.count_nonzero(y_pred * (y_true - 1))


def TN(y_true, y_pred):
    # y_pred = tf.argmax(y_pred, 1)
    # y_true = tf.argmax(y_true, 1)
    return tf.math.count_nonzero((y_pred - 1) * (y_true - 1))


def FN(y_true, y_pred):
    # y_pred = tf.argmax(y_pred, 1)
    # y_true = tf.argmax(y_true, 1)
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


def avg_precision(y_true, y_pred):
    return


def main():
    # Cumulative Training
    # Non-cumulative training
    # Training with just 1 person
    # Training with weighted loss
    # Training with moving training window for balancing

    model = mA.build_model((215, 215, 1), model_type='CNN')
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

    model.compile(optimizer, 'binary_crossentropy', ['accuracy', TP, tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto',
                                                      baseline=None, restore_best_weights=True)

    train_data, train_labels_binary = None, None
    d = {}
    for idx, folder in enumerate(sorted(os.listdir(GROUP_2_PATH))):
        print("Session: {}".format(idx))
        f_path = os.path.join(GROUP_2_PATH, folder)
        df = dL.create_pain_df(f_path)
        # df = df[df['Person'] == 52]
        f_path = df['img_path'].values
        val_data, val_labels_binary, val_labels_people, val_labels = Experiments.load_and_prepare_data(f_path, 0, 4,
                                                                                                       'CNN')
        if idx > 0:
            print("Val_Balance: {:,.0%}".format(np.sum(val_labels_binary[:, 1]) / len(val_labels_binary)))
            history = model.fit(train_data, train_labels_binary, batch_size=32, epochs=30,
                                validation_data=(val_data, val_labels_binary), callbacks=[early_stopping])
            if not d:
                d = history.history
            else:
                for key, val in history.history.items():
                    d[key].extend(val)

        # df = dL.create_pain_df(f_path)
        # print("{}: Actual number of images: ".format(folder), len(df), "thereof pain: ", sum(df['Pain'] != '0'))
        # df = dL.balance_data(df, threshold=200)
        # train_data, train_labels_binary, train_labels_people, train_labels = Experiments.load_and_prepare_data(
        #     df['img_path'].values, 0, 4, 'CNN')
        if idx <= 0:
            train_data, train_labels_binary = val_data, val_labels_binary
        else:
            train_data, train_labels_binary = np.concatenate((train_data, val_data)), np.concatenate(
                (train_labels_binary, val_labels_binary))

    file = os.path.join(RESULTS, 'No_data_balancing.csv')
    pd.DataFrame(d).to_csv(file)

    twilio = Experiments.Twilio()
    twilio.send_message("Training Done")


if __name__ == '__main__':
    main()
