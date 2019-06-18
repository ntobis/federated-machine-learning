# Imports
from __future__ import absolute_import, division, print_function, unicode_literals

import os

import matplotlib.pyplot as plt
import tensorflow as tf

models = tf.keras.models  # like 'from tensorflow.keras import models' (PyCharm import issue workaround)
layers = tf.keras.layers  # like 'from tensorflow.keras import layers' (PyCharm import issue workaround)

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------ Paths ----------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS = os.path.join(ROOT, 'Models')
CENTRALIZED_CHECK_POINT = os.path.join(MODELS, "centralized_cp.ckpt")
CENTRALIZED_MODEL = os.path.join(MODELS, "centralized_model.h5")

# ---------------------------------------------------- End Paths --------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Utility Functions ----------------------------------------------- #


def load_MNIST_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    return train_images, train_labels, test_images, test_labels


def create_checkpoint_callback():
    # Create checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(CENTRALIZED_CHECK_POINT,
                                                     save_weights_only=True,
                                                     verbose=1)
    return cp_callback


# ---------------------------------------------- End Utility Functions --------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# ----------------------------------------------- Model Configuration ---------------------------------------------- #

def build_cnn():
    # Set up model type
    model = models.Sequential()

    # Add layers, inspired by https://www.tensorflow.org/beta/tutorials/images/intro_to_cnns
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train_cnn(model, train_images, train_labels, callbacks):
    model.fit(train_images, train_labels, epochs=2, batch_size=32, validation_split=0.25, callbacks=callbacks)
    return model


def evaluate_cnn(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    return test_loss, test_acc


# --------------------------------------------- End Model Configuration -------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# ----------------------------------------------- Plotting and Display --------------------------------------------- #


def plot_accuracy(model):
    # Plot training & validation accuracy values
    try:  # Depending on the TF version, these are labeled differently
        plt.plot(model.history.history['accuracy'])  # 'accuracy'
        plt.plot(model.history.history['val_accuracy'])
    except KeyError:
        plt.plot(model.history.history['acc'])  # 'acc'
        plt.plot(model.history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def plot_loss(model):
    # Plot training & validation loss values
    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def display_images(train_data, train_labels):
    train_data = tf.reshape(train_data, [60000, 28, 28])
    # Display Digits
    fig = plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(train_data[i], cmap='gray', interpolation='none')
        plt.title("Digit: {}".format(train_labels[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


# --------------------------------------------- End Plotting and Display ------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

# Main function including a number of flags that can be set
def main(plotting=False,
         training=True,
         loading=False,
         evaluating=True,
         max_samples=None):
    # Load data
    train_images, train_labels, test_images, test_labels = load_MNIST_data()

    if max_samples:
        train_images = train_images[:max_samples]
        train_labels = train_labels[:max_samples]

    # Display data
    if plotting:
        display_images(train_images, train_labels)

    # Enable check-pointing
    cp_callback = create_checkpoint_callback()

    # Build model
    model = build_cnn()
    if loading:
        model.load_weights(CENTRALIZED_CHECK_POINT)

    # Train model
    if training:
        model = train_cnn(model, train_images, train_labels, [cp_callback])
        # Save full model
        model.save(CENTRALIZED_MODEL)

    # Evaluate model
    if evaluating:
        test_loss, test_acc = evaluate_cnn(model, test_images, test_labels)
        print("Test Loss: {:5.2f}".format(test_loss))
        print("Test Accuracy: {:5.2f}%".format(100 * test_acc))

    # Plot Accuracy and Loss
    if plotting:
        plot_accuracy(model)
        plot_loss(model)


if __name__ == '__main__':
    main(training=True, loading=False, max_samples=None)
