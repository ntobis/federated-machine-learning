import tensorflow as tf

models = tf.keras.models  # like 'from tensorflow.keras import models' (PyCharm import issue workaround)
layers = tf.keras.layers  # like 'from tensorflow.keras import layers' (PyCharm import issue workaround)


def build_CNN(input_shape):
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

    print("Setting up CNN")
    # Set up model type
    model = models.Sequential(name='CNN')

    # Add layers
    model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), input_shape=input_shape))
    model.add(layers.ReLU())
    model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2)))
    model.add(layers.ReLU())
    model.add(layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2)))
    model.add(layers.MaxPooling2D()),
    model.add(layers.Flatten())
    model.add(layers.Dense(units=128))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dense(units=2, activation='sigmoid'))

    return model


def build_ResNet(input_shape):

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
        layers.Dense(units=2, activation='sigmoid')
    ], name='ResNet')
    return model


def build_model(input_shape, model_type):
    model_type = model_type.lower()
    model_types = {'cnn': build_CNN,
                   'resnet': build_ResNet}
    return model_types[model_type](input_shape=input_shape)


if __name__ == '__main__':
    model_1 = build_model((215, 215, 1), 'CNN')
    model_1.summary()
