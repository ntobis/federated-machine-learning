import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tensorflow as tf
from Scripts import Model_Architectures as mA
from Scripts import Experiments
from Scripts import Data_Loader_Functions as dL

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "Data", "Augmented Data", "Flexible Augmentation")
GROUP_2_PATH = os.path.join(DATA, "group_2")


def main():
    model = mA.build_model((215, 215, 1), model_type='CNN')
    optimizer = tf.keras.optimizers.RMSprop()

    model.compile(optimizer, 'binary_crossentropy', ['accuracy'])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto',
                                                      baseline=None, restore_best_weights=True)
    train_data, train_labels_binary = None, None
    for idx, folder in enumerate(sorted(os.listdir(GROUP_2_PATH))):
        f_path = os.path.join(GROUP_2_PATH, folder)
        val_data, val_labels_binary, val_labels_people, val_labels = Experiments.load_and_prepare_data(f_path, 0, 4,
                                                                                                       'CNN')
        if idx > 0:
            model.fit(train_data, train_labels_binary, batch_size=32, epochs=30,
                      validation_data=(val_data, val_labels_binary), callbacks=[early_stopping])

        # df = dL.create_pain_df(f_path)
        # print("{}: Actual number of images: ".format(folder), len(df), "thereof pain: ", sum(df['Pain'] != '0'))
        # df = dL.balance_data(df, threshold=200)
        # train_data, train_labels_binary, train_labels_people, train_labels = Experiments.load_and_prepare_data(
        #     df['img_path'].values, 0, 4, 'CNN')
        train_data, train_labels_binary, train_labels_people, train_labels = val_data, val_labels_binary, val_labels_people, val_labels


if __name__ == '__main__':
    main()
