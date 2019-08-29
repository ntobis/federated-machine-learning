import tensorflow as tf
from tensorflow.python.keras.metrics import TruePositives, TrueNegatives, FalsePositives, FalseNegatives, Recall, \
    Precision, AUC

models = tf.keras.models  # like 'from tensorflow.keras import models' (PyCharm import issue workaround)
layers = tf.keras.layers  # like 'from tensorflow.keras import layers' (PyCharm import issue workaround)


def build_CNN(input_shape):
    """
    Return a simple CNN model for image classification.

    :param input_shape:     image input shape (tuple), e.g. (28, 28, 1)

    :return:
        model               compiled tensorflow model
    """

    print("Setting up CNN")
    # Set up model type
    model = models.Sequential(name='CNN')

    # Add layers
    model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), input_shape=input_shape, padding='same', strides=(2, 2),
                            name='conv2d_0_global'))
    model.add(layers.BatchNormalization(name='batch_norm_0_global'))
    model.add(layers.ReLU(name='relu_0_global'))
    # model.add(layers.MaxPooling2D(name='max_pool_0_global'))

    model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same', strides=(2, 2), name='conv2d_1_global'))
    model.add(layers.BatchNormalization(name='batch_norm_1_global'))
    model.add(layers.ReLU(name='relu_1_global'))
    # model.add(layers.MaxPooling2D(name='max_pool_1_global'))

    model.add(layers.Conv2D(filters=128, kernel_size=(5, 5), padding='same', strides=(2, 2), name='conv2d_2_global'))
    model.add(layers.BatchNormalization(name='batch_norm_2_global'))
    model.add(layers.ReLU(name='relu_2_global'))
    model.add(layers.MaxPooling2D(name='max_pool_2_global'))

    model.add(layers.Flatten(name='flatten_0_local'))
    model.add(layers.Dense(units=128, name='dense_0_local'))
    model.add(layers.BatchNormalization(name='batch_norm_3_local'))
    model.add(layers.ReLU(name='relu_3_local'))
    model.add(layers.Dense(units=1, activation='sigmoid', name='dense_1_local'))

    return model


def build_ResNet(input_shape):
    """
    Return a tensorflow model with ResNet 50 as teh feature extractor and two dense layers with Relu and Sigmoid
    activation respectively as classification layers.

    :param input_shape:          image input shape (tuple), e.g. (28, 28, 3)
    :return:
        model                    Tensorflow model
    """

    print("Setting up ResNet")
    base_model = tf.keras.applications.ResNet50(include_top=False, input_shape=(input_shape[0], input_shape[1], 3),
                                                weights='imagenet')
    # Freeze the pre-trained model weights
    base_model.trainable = False

    # Layer classification head with feature detector
    model = tf.keras.Sequential([
        base_model,
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(units=128),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dense(units=1, activation='sigmoid')
    ], name='ResNet')
    return model


def build_model(input_shape, model_type):
    model_type = model_type.lower()
    model_types = {'cnn': build_CNN,
                   'resnet': build_ResNet}
    return model_types[model_type](input_shape=input_shape)


if __name__ == '__main__':
    model_test = build_model((215, 215, 1), 'CNN')
    model_test.summary(line_length=100)
