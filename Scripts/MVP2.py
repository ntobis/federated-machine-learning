import os

import tensorflow as tf

from Scripts import Model_Architectures as mA, Experiments, Keras_Custom as kC

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "Data", "Augmented Data", "Flexible Augmentation")
GROUP_2_PATH = os.path.join(DATA, "group_2")
RESULTS = os.path.join(ROOT, "Results")
from Scripts import Data_Loader_Functions as dL


def main():
    model_type = 'CNN'
    model = mA.build_model((215, 215, 1), model_type=model_type)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(optimizer, 'binary_crossentropy', ['accuracy',
                                                     kC.TP, kC.TN, kC.FP, kC.FN
                                                     ])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto',
                                                      baseline=None, restore_best_weights=True)

    f_path = os.path.join(GROUP_2_PATH, 'session_1')
    df = dL.create_pain_df(f_path)[:400]
    val_data1, val_labels_binary1, _, _ = Experiments.load_and_prepare_data(df['img_path'].values, 0, 4, model_type)

    f_path = os.path.join(GROUP_2_PATH, 'session_2')
    df = dL.create_pain_df(f_path)[:400]
    val_data2, val_labels_binary2, _, _ = Experiments.load_and_prepare_data(df['img_path'].values, 0, 4, model_type)

    f_path = os.path.join(GROUP_2_PATH, 'session_3')
    df = dL.create_pain_df(f_path)[:400]
    train_data, train_labels_binary, _, _ = Experiments.load_and_prepare_data(df['img_path'].values, 0, 4, model_type)

    history = kC.AdditionalValidationSets([(val_data2, val_labels_binary2, 'val2'),
                                           ])

    model.fit(train_data, train_labels_binary, batch_size=32, epochs=5, callbacks=[early_stopping,
                                                                                   history],
              validation_data=(val_data1, val_labels_binary1))


if __name__ == '__main__':
    main()
    # print(kC.FN([1,1],[2,2]).name)
