from __future__ import print_function

import math
import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------ Paths ----------------------------------------------------- #

ROOT = os.path.dirname(os.path.dirname(__file__))
FIGURES = os.path.join(ROOT, "Figures")
RESULTS = os.path.join(ROOT, "Results")


# ---------------------------------------------------- End Paths --------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------------------------ #
# --------------------------------------------------- Parameters --------------------------------------------------- #

class PlotParams:
    def __init__(self, dataset, experiment, metric='Accuracy', title='', x_label='', y_label='',
                 legend_loc='upper left',
                 num_format="{:5.1f}%", max_epochs=None, label_spaces=1, suffix=''):
        self.metric = metric
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.legend_loc = legend_loc
        self.num_format = num_format
        self.dataset = dataset
        self.experiment = experiment
        self.max_epochs = max_epochs
        self.label_spaces = label_spaces
        self.suffix = suffix
        self.colors = ['#CD6155', '#EC7063', '#AF7AC5', '#A569BD', '#5499C7', '#5DADE2', '#48C9B0', '#45B39D',
                       '#52BE80', '#58D68D', '#F4D03F', '#F5B041', '#F5B041', '#DC7633']


# ------------------------------------------------- End Parameters ------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------- Print Functions ------------------------------------------------ #

def print_communication_round(com_round):
    print()
    print("-" * 131)
    print("{} Communication Round {} {}".format("-" * math.floor((130 - 21 - len(str(com_round))) / 2), com_round,
                                                "-" * math.ceil((130 - 21 - len(str(com_round))) / 2)))


def print_client_id(client_id):
    print()
    print("{} Client {} {}".format("-" * math.floor((130 - 8 - len(str(client_id))) / 2), client_id,
                                   "-" * math.ceil((130 - 8 - len(str(client_id))) / 2)))


def print_loss_accuracy(accuracy, loss, data_type="Test"):
    print("-----------------------")
    print("{} Loss: {:5.2f}".format(data_type, loss))
    print("{} Accuracy: {:5.2f}%".format(data_type, 100 * accuracy))
    print("-----------------------")
    print()


def print_shard(percentage):
    print("\n\n\033[1m{} Shard {:.0%} {}\033[0m".format("-" * math.floor((130 - 7 - len(str(percentage))) / 2),
                                                        percentage,
                                                        "-" * math.ceil((130 - 7 - len(str(percentage))) / 2)))


def print_experiment(experiment):
    print("\n\n\033[1m{} Experiment {} {}\033[0m".format("-" * math.floor((130 - 12 - len(experiment)) / 2), experiment,
                                                         "-" * math.ceil((130 - 12 - len(experiment)) / 2)))


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
    plt.xticks(np.arange(min(history.index + 1), max(history.index + 1) + 1, step=1))
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
    dataset = params.dataset
    experiment = params.experiment
    max_epochs = params.max_epochs
    label_spaces = params.label_spaces
    suffix = params.suffix

    # Plot line
    try:
        plt.plot((history.index + 1)[:max_epochs], history['Centralized {}'.format(metric)][:max_epochs],
                 color=colors[0])

        # Plot labels
        for i, j in history['Centralized {}'.format(metric)][:max_epochs].items():
            if not int(i) % label_spaces:
                plt.text((i + 1) * 0.99, j, num_format.format(j), color='black',
                         bbox=dict(facecolor='white', edgecolor=colors[0], boxstyle='round'))
    except KeyError:
        pass

    # Get federated lines
    federated_accuracy_cols = [str(col) for col in history.columns if 'Federated Test {}'.format(metric) in col]
    for idx, col in enumerate(federated_accuracy_cols):

        # Plot lines
        plt.plot((history.index + 1)[:max_epochs], history[col][:max_epochs], color=colors[idx + 1])

        # Plot labels
        for i, j in history[col][:max_epochs].items():
            if not int(i) % label_spaces:
                plt.text((i + 1) * 0.99, j, num_format.format(j), color='black',
                         bbox=dict(facecolor='white', edgecolor=colors[idx + 1], boxstyle='round'))

    # Draw graph
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xticks(np.arange(min(history.index + 1), max((history.index + 1)[:max_epochs]) + 1, step=1))
    federated_accuracy_cols.insert(0, "Centralized {}".format(metric))  # Add centralized to list of legend labels
    plt.legend(federated_accuracy_cols, loc=legend_loc)
    file = time.strftime("%Y-%m-%d-%H%M%S") + r"_{}_{}_{}_{}.png".format(dataset, experiment, metric, suffix)
    fig = plt.gcf()
    fig.set_size_inches((12, 8), forward=False)
    plt.savefig(os.path.join(FIGURES, file), dpi=300)
    plt.show()
    plt.clf()


def display_images(train_data, train_labels):
    """
    Display first 9 MNIST images

    :param train_data:      numpy array of shape (60000, 28, 28, 1)
    :param train_labels:    numpy array of shape (60000, )

    """

    train_data = tf.reshape(train_data, [train_data.shape[0], 28, 28])
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


def make_pain_plot(folder, params, final_epoch=29):
    files = os.listdir(folder)
    files = [os.path.join(folder, file) for file in files if file.endswith('{}.csv'.format(final_epoch))]
    files = sorted(files)

    legend = []
    for idx, file in enumerate(files):
        df = pd.read_csv(file)
        plt.plot(df[params.metric], color=params.colors[idx + 1])
        for i, j in df[params.metric][:params.max_epochs].items():
            if not int(i) % params.label_spaces:
                plt.text(i, j, params.num_format.format(j), color='black',
                         bbox=dict(facecolor='white', edgecolor=params.colors[idx + 1], boxstyle='round'))

        legend.append('Group 1 + {0:.0%}% Group 2'.format(0.1 * idx))

    plt.legend(legend, loc=params.legend_loc)
    plt.title('{} | {} | Group 1 + X% Group 2'.format(params.metric, params.experiment))
    plt.yticks(np.arange(0.3, 1.05, step=0.05))
    plt.ylabel('{}'.format(params.metric))
    plt.xlabel('Epochs / Communication Rounds')
    return plt


def make_pain_plot_grid(folder, metrics, params, final_epoch=29):
    for idx, metric in enumerate(metrics):
        params.metric = metric
        plt.subplot(2, 2, idx + 1)
        make_pain_plot(folder, params=params, final_epoch=final_epoch)

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    file = time.strftime("%Y-%m-%d-%H%M%S") + r"_{}_{}_{}.png".format(params.dataset, params.experiment, str(metrics))
    fig = plt.gcf()
    fig.suptitle('Group 2 | 40% Test Set | Evaluation', fontsize=20)
    fig.set_size_inches((24, 16), forward=False)
    plt.savefig(os.path.join(FIGURES, file), dpi=300)
    plt.show()


# ----------------------------------------------- End Plot Functions ----------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

if __name__ == '__main__':
    parameters = PlotParams(
        dataset='Pain',
        experiment='Centralized',
        metric='F1_Score',
        legend_loc='lower right',
        num_format="{:5.1%}",
        max_epochs=None,
        label_spaces=4
    )
    rep_metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    make_pain_plot_grid(RESULTS, rep_metrics, parameters)


def print_shard_summary(labels, people):
    print("Pain:     ", int(np.sum(labels[:, 1])))
    print("No Pain:  ", int(len(labels) - np.sum(labels[:, 1])))
    print("Total:    ", len(labels))
    print("Ratio:     {:.1%}".format(int(np.sum(labels[:, 1])) / len(labels)))
    print("People:   ", np.unique(people, return_counts=True)[0])
    print("People(N):", np.unique(people, return_counts=True)[1])
