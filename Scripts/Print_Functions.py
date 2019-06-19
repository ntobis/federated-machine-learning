from __future__ import print_function

import sys

import tensorflow as tf
from matplotlib import pyplot as plt


def print_communication_round(com_round):
    print()
    print("----------------------------------------------------------------------------------------------------------------------------------------")
    print("-------------------------------------------------------- Communication Round {} --------------------------------------------------------".format(com_round))


def print_client_id(id):
    print()
    print("----------------------------------------------------------------------------------------------------------------------------------------")
    print("--------------------------------------------------------------- Client {} --------------------------------------------------------------".format(id))


def print_loss_accuracy(test_acc, test_loss):
    print("-----------------------")
    print("Test Loss: {:5.2f}".format(test_loss))
    print("Test Accuracy: {:5.2f}%".format(100 * test_acc))
    print("-----------------------")
    print()


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def plot_accuracy(model):
    """
    Plot training & validation accuracy values over epochs

    :param model:           trained tensorflow model holding a 'History' objects

    """

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


def plot_federated_accuracy(history):
    """
    Plot accuracy of federated learning over communication rounds

    :param history:           pandas data-frame with 'Accuracy' column

    """

    plt.plot(history['Accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Communication Round')
    plt.legend(['Test'], loc='upper left')
    plt.show()


def plot_loss(model):
    """
    Plot training & validation loss values over epochs

    :param model:           trained tensorflow model holding a 'History' objects

    """

    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def plot_federated_loss(history):
    """
    Plot loss of federated learning over communication rounds

    :param history:           pandas data-frame with 'Loss' column

    """

    plt.plot(history['Loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Communication Round')
    plt.legend(['Test'], loc='upper left')
    plt.show()


def display_images(train_data, train_labels):
    """
    Display first 9 MNIST images

    :param train_data:      numpy array of shape (60000, 28, 28, 1)
    :param train_labels:    numpy array of shape (60000, )

    """

    train_data = tf.reshape(train_data, [60000, 28, 28])
    # Display Digits
    plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(train_data[i], cmap='gray', interpolation='none')
        plt.title("Digit: {}".format(train_labels[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()