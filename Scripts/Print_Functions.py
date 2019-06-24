from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


# ------------------------------------------------------------------------------------------------------------------ #
# --------------------------------------------------- Parameters --------------------------------------------------- #


class PlotParams:
    def __init__(self, metric='Accuracy', title='', x_label='', y_label='', legend_loc='upper left',
                 num_format="{:5.1f}%"):
        self.metric = metric
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.legend_loc = legend_loc
        self.num_format = num_format
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


# ------------------------------------------------- End Parameters ------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------- Print Functions ------------------------------------------------ #

def print_communication_round(com_round):
    print()
    print("------------------------------------------------------------------------------------------------------------"
          "----------------------------")
    print("-------------------------------------------------------- Communication Round {} ----------------------------"
          "----------------------------".format(com_round))


def print_client_id(client_id):
    print()
    print("------------------------------------------------------------------------------------------------------------"
          "----------------------------")
    print("--------------------------------------------------------------- Client {} ----------------------------------"
          "----------------------------".format(client_id))


def print_loss_accuracy(accuracy, loss, data_type="Test"):
    print("-----------------------")
    print("{} Loss: {:5.2f}".format(data_type, loss))
    print("{} Accuracy: {:5.2f}%".format(data_type, 100 * accuracy))
    print("-----------------------")
    print()


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# ----------------------------------------------- End Print Functions ---------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------- Plot Functions ------------------------------------------------- #


def plot_centralized_accuracy(model):
    """
    Plot training & validation accuracy values over epochs

    :param model:           trained tensorflow model holding a 'History' objects

    """

    try:  # Depending on the TF version, these are labeled differently
        plt.plot(model.history.history['accuracy'])  # 'accuracy'
        # plt.plot(model.history.history['val_accuracy'])

    except KeyError:
        plt.plot(model.history.history['acc'])  # 'acc'
        # plt.plot(model.history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def plot_centralized_loss(model):
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


def plot_federated_accuracy(history):
    """
    Plot accuracy of federated learning over communication rounds

    :param history:           pandas data-frame with 'Accuracy' column

    """

    plt.plot(history.index + 1, history['Train Accuracy'])
    plt.plot(history.index + 1, history['Test Accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Communication Round')
    plt.xticks(np.arange(min(history.index+1), max(history.index + 1) + 1, step=1))
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def plot_federated_loss(history):
    """
    Plot loss of federated learning over communication rounds

    :param history:           pandas data-frame with 'Loss' column

    """

    plt.plot(history.index + 1, history['Train Loss'])
    plt.plot(history.index + 1, history['Test Loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Communication Round')
    plt.xticks(np.arange(min(history.index + 1), max(history.index + 1) + 1, step=1))
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def plot_joint_metric(history, params):
    """
    Plot a specified metric for federated and centralized learning

    :param history:           pandas data-frame with 'Accuracy' column
    :param params:            parameter object, with fields to plot a metric

    """
    # Get parameters
    colors = params.colors
    metric = params.metric
    title = params.title
    x_label = params.x_label
    y_label = params.y_label
    legend_loc = params.legend_loc
    num_format = params.num_format

    # Plot line
    plt.plot(history.index + 1, history['Centralized {}'.format(metric)], color=colors[0])

    # Plot labels
    for i, j in history['Centralized {}'.format(metric)].items():
        plt.text((i + 1) * 0.99, j * 1.02, num_format.format(j), color='black',
                 bbox=dict(facecolor='white', edgecolor=colors[0], boxstyle='round'))

    # Get federated lines
    federated_accuracy_cols = [str(col) for col in history.columns if 'Federated Test {}'.format(metric) in col]
    for idx, col in enumerate(federated_accuracy_cols):

        # Plot lines
        plt.plot(history.index + 1, history[col], color=colors[idx+1])

        # Plot labels
        for i, j in history[col].items():
            plt.text((i + 1) * 0.99, j * 0.98, num_format.format(j), color='black',
                     bbox=dict(facecolor='white', edgecolor=colors[idx+1], boxstyle='round'))

    # Draw graph
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xticks(np.arange(min(history.index+1), max(history.index + 1) + 1, step=1))
    federated_accuracy_cols.insert(0, "Centralized {}".format(metric))  # Add centralized to list of legend labels
    plt.legend(federated_accuracy_cols, loc=legend_loc)
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


# ----------------------------------------------- End Plot Functions ----------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #
