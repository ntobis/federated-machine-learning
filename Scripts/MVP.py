import os
import sys

import pandas as pd

from Scripts.Keras_Custom import TP, FP, TN, FN

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tensorflow as tf
import numpy as np
from Scripts import Model_Architectures as mA
from Scripts import Experiments
from Scripts import Data_Loader_Functions as dL


ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "Data", "Augmented Data", "Flexible Augmentation")
GROUP_2_PATH = os.path.join(DATA, "group_2")
RESULTS = os.path.join(ROOT, "Results")


def main():
    twilio = Experiments.Twilio()

    model_type = 'CNN'
    model = mA.build_model((215, 215, 1), model_type=model_type)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(optimizer, 'binary_crossentropy', ['accuracy', TP, TN, FP, FN])


    # for person in [43, 48, 52, 59, 64, 80, 92, 96, 107, 109, 115, 120]:
    train_data, train_labels_binary = None, None
    d = None
    train_df = dL.create_pain_df(GROUP_2_PATH)[:40]
    for idx, folder in enumerate(sorted(os.listdir(GROUP_2_PATH))):
        print("Session: {}".format(idx))
        f_path = os.path.join(GROUP_2_PATH, folder)
        df = dL.create_pain_df(f_path)[:40]
        # df = df[df['Person'] == person]
        f_paths = df['img_path'].values
        if len(f_paths) > 0:
            val_data, val_labels_binary, val_labels_people, val_labels = Experiments.load_and_prepare_data(
                f_paths,
                0, 4, model_type)

            if idx > 0 and train_data is not None:
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

            df = train_df[(train_df['Session'] <= idx)
                          # & (train_df['Person'] == person)
                          ]
            print("{}: Actual number of images: ".format(folder), len(df), "thereof pain: ", sum(df['Pain'] != '0'))
            # df = dL.balance_data(df, threshold=200)
            if len(df) > 0:
                train_data, train_labels_binary, train_labels_people, train_labels = Experiments.load_and_prepare_data(
                    df['img_path'].values, 0, 4, model_type)
            else:
                train_data, train_labels_binary, train_labels_people, train_labels = [None] * 4
            # if idx <= 0:
            #     train_data, train_labels_binary = val_data, val_labels_binary
            # else:
            #     train_data, train_labels_binary = np.concatenate((train_data, val_data)), np.concatenate(
            #         (train_labels_binary, val_labels_binary))

        # file = os.path.join(RESULTS, 'CNN Individual Training_Validation small LR Balancing Person  {}.csv'.format(person))
        file = os.path.join(RESULTS, 'TEST.csv')
        if d is not None:
            d.to_csv(file)
        # del model
        # print("Initializing new model")
        # model = mA.build_model((215, 215, 1), model_type=model_type)
        # optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
        # optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

        # model.compile(optimizer, 'binary_crossentropy', ['accuracy', TP, TN, FP, FN])

    twilio.send_message("Training Done")


if __name__ == '__main__':
    main()
