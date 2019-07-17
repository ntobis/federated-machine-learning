import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time

import pandas as pd
import numpy as np

from Scripts import Centralized_CNN as cNN
from Scripts import Federated_CNN as fed_CNN
from Scripts import Differentially_Private_CNN as diff_CNN
from Scripts import Print_Functions as Output
from Scripts import Data_Loader_Functions as Data_Loader

pd.set_option('display.max_columns', 500)


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Utility Functions ----------------------------------------------- #


def move_results(experiment, date, keys):
    """
    Utility function used to move results into a specified folder with a name of the format:
    EXPERIMENT DATE KEYS

    :param experiment:                  string, name of the experiment
    :param date:                        string, date of the experiment
    :param keys:                        list, type of experiment, e.g. number of clients: [2, 5, 10, 100]
    :return:
    """

    experiment_path = os.path.join(cNN.RESULTS, experiment + " " + date + " " + str(keys))
    if not os.path.isdir(experiment_path):
        os.mkdir(experiment_path)
    for elem in os.listdir(cNN.RESULTS):
        if os.path.isfile(os.path.join(cNN.RESULTS, elem)):
            old_loc = os.path.join(cNN.RESULTS, elem)
            new_loc = os.path.join(cNN.RESULTS, experiment_path, elem)
            os.rename(old_loc, new_loc)


def combine_results(experiment, keys, sub_folder=None):
    """
    Combines the results from various experiments into one data frame that can be used for joint plotting of metrics.

    :param experiment:                  string, the type of experimental setting to be used, e.g. "CLIENTS"
    :param keys:                        array, the different experiments, e.g. number of clients [2, 5, 10]
    :param sub_folder:                  string, where in the "Results" folder the results should be stored

    :return:
        history:                        DataFrame, holds the results of all experiments, passed to a plotting function
    """

    # Open most recent history files
    if sub_folder:
        files = os.listdir(os.path.join(cNN.RESULTS, sub_folder))
        files = [os.path.join(os.path.join(cNN.RESULTS, sub_folder), file) for file in files]
    else:
        files = os.listdir(cNN.RESULTS)
        files = [os.path.join(cNN.RESULTS, file) for file in files]

    # Combine Results
    sorted_files = sorted(files, key=os.path.getctime)
    centralized = pd.read_csv(sorted_files[0], index_col=0)
    history = centralized
    for idx in range(len(keys)):
        federated = pd.read_csv(sorted_files[-idx - 1], index_col=0)
        federated = federated.rename(index=str,
                                     columns={"Federated Train Loss": "Federated Train Loss {} {}".format(
                                         experiment,
                                         keys[idx]),
                                         "Federated Train Accuracy": "Federated Train Accuracy {} {}".format(
                                             experiment,
                                             keys[idx]),
                                         "Federated Test Loss": "Federated Test Loss {} {}".format(
                                             experiment,
                                             keys[idx]),
                                         "Federated Test Accuracy": "Federated Test Accuracy {} {}".format(
                                             experiment,
                                             keys[idx])})
        history = pd.concat([history.reset_index(drop=True), federated.reset_index(drop=True)], axis=1)

    return history


# ---------------------------------------------- End Utility Functions --------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Experiment Runners ---------------------------------------------- #


def plot_results(dataset, experiment, keys, date, suffix, move=False):
    """
    Sets the parameters for the plotting function, and calls the plotting function to plott loss and accuracy over
    multiple epochs/communication rounds.

    :param dataset:                 string, name of the dataset to be used, e.g. "MNIST"
    :param experiment:              string, the type of experimental setting to be used, e.g. "CLIENTS"
    :param keys:                    array, the different experiments, e.g. number of clients [2, 5, 10]
    :param date:                    string, date to be used for folder naming
    :param suffix:                  string, additional information to be added to the folder name
    :param move:                    bool, set to true if results are still in general "Results" folder
    :return:
    """

    if move:
        move_results(experiment, time.strftime("%Y-%m-%d"), keys)

    sub_folder = experiment + " " + date + " " + suffix
    history = combine_results(experiment, keys, sub_folder)

    # Plot Accuracy
    params = Output.PlotParams(
        dataset=dataset,
        experiment=experiment,
        metric='Accuracy',
        title='Model Accuracy {} {}'.format(experiment, suffix),
        x_label='Federated Comm. Round/Centralized Epoch',
        y_label='Accuracy',
        legend_loc='lower right',
        num_format="{:5.1%}",
        max_epochs=None,
        label_spaces=4,
        suffix=suffix
    )
    Output.plot_joint_metric(history, params)

    # Plot Loss
    params = Output.PlotParams(
        dataset=dataset,
        experiment=experiment,
        metric='Loss',
        title='Model Loss {} {}'.format(experiment, suffix),
        x_label='Federated Comm. Round/Centralized Epoch',
        y_label='Loss',
        legend_loc='upper right',
        num_format="{:5.2f}",
        max_epochs=None,
        label_spaces=4,
        suffix=suffix
    )
    Output.plot_joint_metric(history, params)


def experiment_centralized(dataset, experiment, train_data, train_labels, test_data, test_labels, epochs=5):
    """
    Sets up a centralized CNN that trains on a specified dataset. Saves the results to CSV.

    :param dataset:                 string, name of the dataset to be used, e.g. "MNIST"
    :param experiment:              string, the type of experimental setting to be used, e.g. "CLIENTS"
    :param train_data:              numpy array, the train data
    :param train_labels:            numpy array, the train labels
    :param test_data:               numpy array, the test data
    :param test_labels:             numpy array, the test labels
    :param epochs:                  int, number of epochs that the centralized CNN trains for
    :return:
    """

    # Train Centralized CNN
    centralized_model = cNN.build_cnn(input_shape=(28, 28, 1))
    centralized_model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    centralized_model = cNN.train_cnn(centralized_model, train_data, train_labels, epochs=epochs)

    # Save full model
    centralized_model.save(cNN.CENTRALIZED_MODEL)

    # Evaluate model
    test_loss, test_acc = cNN.evaluate_cnn(centralized_model, test_data, test_labels)
    Output.print_loss_accuracy(test_acc, test_loss)

    # Save results
    history = pd.DataFrame.from_dict(centralized_model.history.history)
    history = history.rename(index=str, columns={"loss": "Centralized Loss", "accuracy": "Centralized Accuracy"})
    file = time.strftime("%Y-%m-%d-%H%M%S") + r"_Centralized_{}_{}.csv".format(dataset, experiment)
    history.to_csv(os.path.join(cNN.RESULTS, file))


def experiment_federated(clients, dataset, experiment, train_data, train_labels, test_data, test_labels,
                         rounds=5, epochs=1, split='random', participants=None):
    """
    Sets up a federated CNN that trains on a specified dataset. Saves the results to CSV.

    :param clients:                 int, the maximum number of clients participating in a communication round
    :param dataset:                 string, name of the dataset to be used, e.g. "MNIST"
    :param experiment:              string, the type of experimental setting to be used, e.g. "CLIENTS"
    :param train_data:              numpy array, the train data
    :param train_labels:            numpy array, the train labels
    :param test_data:               numpy array, the test data
    :param test_labels:             numpy array, the test labels
    :param rounds:                  int, number of communication rounds that the federated clients average results for
    :param epochs:                  int, number of epochs that the client CNN trains for
    :param split:                   Determine if split should occur randomly
    :param participants:            participants in a given communications round
    :return:
    """

    train_data, train_labels = Data_Loader.split_data_into_clients(clients, split, train_data, train_labels)

    # Reset federated model
    fed_CNN.reset_federated_model()

    # Train Model
    history = fed_CNN.federated_learning(communication_rounds=rounds,
                                         num_of_clients=clients,
                                         train_data=train_data,
                                         train_labels=train_labels,
                                         test_data=test_data,
                                         test_labels=test_labels,
                                         epochs=epochs,
                                         num_participating_clients=participants
                                         )

    # Save history for plotting
    file = time.strftime("%Y-%m-%d-%H%M%S") + r"_Federated_{}_{}_rounds_{}_clients_{}.csv".format(dataset, experiment,
                                                                                                  rounds, clients)
    history = history.rename(index=str, columns={"Train Loss": "Federated Train Loss",
                                                 "Train Accuracy": "Federated Train Accuracy",
                                                 "Test Loss": "Federated Test Loss",
                                                 "Test Accuracy": "Federated Test Accuracy"})
    history.to_csv(os.path.join(cNN.RESULTS, file))


def experiment_differential_privacy(clients, dataset, experiment, train_data, train_labels, test_data, test_labels,
                                    sigma, rounds=5, epochs=1, split='random', participants=None, learning_rate=0.01):
    """
    Sets up a federated CNN that trains on a specified dataset. Saves the results to CSV.

    :param clients:                 int, the maximum number of clients participating in a communication round
    :param dataset:                 string, name of the dataset to be used, e.g. "MNIST"
    :param experiment:              string, the type of experimental setting to be used, e.g. "CLIENTS"
    :param train_data:              numpy array, the train data
    :param train_labels:            numpy array, the train labels
    :param test_data:               numpy array, the test data
    :param test_labels:             numpy array, the test labels
    :param sigma:                   float, determining the level of differential privacy
    :param rounds:                  int, number of communication rounds that the federated clients average results for
    :param epochs:                  int, number of epochs that the client CNN trains for
    :param split:                   Determine if split should occur randomly
    :param participants:            participants in a given communications round
    :type learning_rate:            float, determining the learning rate for the training algorithm
    :return:
    """

    train_data, train_labels = Data_Loader.split_data_into_clients(clients, split, train_data, train_labels)

    # Reset federated model
    fed_CNN.reset_federated_model()

    # Train Model
    history = diff_CNN.federated_learning(communication_rounds=rounds,
                                          num_of_clients=clients,
                                          train_data=train_data,
                                          train_labels=train_labels,
                                          test_data=test_data,
                                          test_labels=test_labels,
                                          epochs=epochs,
                                          sigma=sigma,
                                          num_participating_clients=participants,
                                          learning_rate=learning_rate
                                          )

    # Save history for plotting
    file = time.strftime("%Y-%m-%d-%H%M%S") + r"_Federated_{}_{}_rounds_{}_clients_{}.csv".format(dataset, experiment,
                                                                                                  rounds, clients)
    history = history.rename(index=str, columns={"Train Loss": "Federated Train Loss",
                                                 "Train Accuracy": "Federated Train Accuracy",
                                                 "Test Loss": "Federated Test Loss",
                                                 "Test Accuracy": "Federated Test Accuracy"})
    history.to_csv(os.path.join(cNN.RESULTS, file))


# ---------------------------------------------- End Experiment Runners -------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# -------------------------------------------------- Experiments - 1 ----------------------------------------------- #


def experiment_1_number_of_clients(dataset, experiment, rounds, clients):
    """
    First experiment conducted. Experimenting with varying the number of clients used in a federated setting.

    :param dataset:                 string, name of the dataset to be used, e.g. "MNIST"
    :param experiment:              string, the type of experimental setting to be used, e.g. "CLIENTS"
    :param rounds:                  int, number of communication rounds that the federated clients average results for
    :param clients:                 int_array, the maximum number of clients participating in a communication round
    :return:
    """

    # Load data
    train_data, train_labels, test_data, test_labels, dataset = Data_Loader.load_data(dataset)

    # Perform Experiments
    for client_num in clients:
        experiment_federated(client_num, dataset, experiment, train_data, train_labels, test_data, test_labels, rounds)

    experiment_centralized(dataset, experiment, train_data, train_labels, test_data, test_labels, rounds)


def experiment_2_limited_digits(dataset, experiment, rounds, digit_array):
    """
    Second experiment conducted. Experimenting with varying the number of MNIST digits used in a centralized and
    federated setting. The number of clients is held constant at 10.

    :param dataset:                 string, name of the dataset to be used, e.g. "MNIST"
    :param experiment:              string, the type of experimental setting to be used, e.g. "CLIENTS"
    :param rounds:                  int, number of communication rounds that the federated clients average results for
    :param digit_array:             array of arrays, specifying the digits to be used in a given experiment,
                                    e.g. [[0,5],[0,2,5]]
    :return:
    """

    # Load data
    train_data, train_labels, test_data, test_labels, dataset = Data_Loader.load_data(dataset)

    # Perform Experiments
    for digit in digit_array:
        experiment = experiment + "_" + str(digit)
        train_data_filtered = train_data[np.in1d(train_labels, digit)]
        train_labels_filtered = train_labels[np.in1d(train_labels, digit)]
        test_data_filtered = test_data[np.in1d(test_labels, digit)]
        test_labels_filtered = test_labels[np.in1d(test_labels, digit)]

        experiment_federated(10, dataset, experiment, train_data_filtered, train_labels_filtered, test_data_filtered,
                             test_labels_filtered, rounds)

        experiment_centralized(dataset, experiment, train_data_filtered, train_labels_filtered, test_data_filtered,
                               test_labels_filtered, rounds)


def experiment_3_add_noise(dataset, experiment, rounds, std_devs):
    """
    Third experiment conducted. Experimenting with varying the noise added to the MNIST data in a centralized and a
    federated setting. The number of clients is held constant at 10.

    :param dataset:                 string, name of the dataset to be used, e.g. "MNIST"
    :param experiment:              string, the type of experimental setting to be used, e.g. "CLIENTS"
    :param rounds:                  int, number of communication rounds that the federated clients average results for
    :param std_devs:                float_array, the standard deviation of gaussian noise to be added to the dataset
    :return:
    """

    # Load data
    train_data, train_labels, test_data, test_labels, dataset = Data_Loader.load_data(dataset)

    # Perform Experiments
    for std_dv in std_devs:
        this_experiment = experiment + "_" + str(std_dv)
        train_data_noise = train_data + np.random.normal(loc=0, scale=std_dv, size=train_data.shape)
        Output.display_images(train_data_noise, train_labels)
        experiment_federated(10, dataset, this_experiment, train_data_noise, train_labels, test_data, test_labels,
                             rounds)

        experiment_centralized(dataset, this_experiment, train_data_noise, train_labels, test_data, test_labels, rounds)


# ------------------------------------------------ End Experiments - 1 --------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

def experiment_main_1():
    """
    Main function running the first 3 experiments.
    :return:
    """

    # Experiment 1 - Number of clients
    clients = [2, 5, 10, 20, 50, 100]
    experiment_1_number_of_clients(dataset="MNIST", experiment="CLIENTS", rounds=30, clients=clients)

    # Plot results
    plot_results(dataset="MNIST", experiment="CLIENTS", keys=clients, date="2019-06-25",
                 suffix=str(clients))

    # Experiment 2 - Digits
    digits_arr = [
        [0, 5],
        [0, 2, 5, 9],
        [0, 2, 4, 5, 7, 9],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ]
    experiment_2_limited_digits(dataset="MNIST", experiment="DIGITS", rounds=30, digit_array=digits_arr)

    # Plot results
    for digits in digits_arr:
        plot_results(dataset="MNIST", experiment="DIGITS", keys=[digits], date="2019-06-26",
                     suffix=str(digits))

    # Experiment 3 - Adding Noise
    std_dev_arr = [0.1, 0.25, 0.5]
    experiment_3_add_noise(dataset="MNIST", experiment="NOISE", rounds=30, std_devs=std_dev_arr)

    # Plot results
    for std_dev in std_dev_arr:
        plot_results(dataset="MNIST", experiment="NOISE", keys=[std_dev], date="2019-06-26",
                     suffix=str(std_dev))


# ------------------------------------------------------------------------------------------------------------------ #
# -------------------------------------------------- Experiments - 2 ----------------------------------------------- #

def experiment_4_split_digits(dataset, experiment, rounds, clients):
    """
    First experiment conducted. Experimenting with varying the number of clients used in a federated setting.

    :param dataset:                 string, name of the dataset to be used, e.g. "MNIST"
    :param experiment:              string, the type of experimental setting to be used, e.g. "CLIENTS"
    :param rounds:                  int, number of communication rounds that the federated clients average results for
    :param clients:                 int_array, the maximum number of clients participating in a communication round
    :return:
    """

    # Load data
    train_data, train_labels, test_data, test_labels, dataset = Data_Loader.load_data(dataset)

    # Perform Experiments
    for client_num in clients:
        experiment_federated(client_num, dataset, experiment, train_data, train_labels, test_data, test_labels, rounds,
                             split='no_overlap')

    experiment_centralized(dataset, experiment, train_data, train_labels, test_data, test_labels, rounds)


def experiment_5_split_digits_with_overlap(dataset, experiment, rounds, clients):
    """
    First experiment conducted. Experimenting with varying the number of clients used in a federated setting.

    :param dataset:                 string, name of the dataset to be used, e.g. "MNIST"
    :param experiment:              string, the type of experimental setting to be used, e.g. "CLIENTS"
    :param rounds:                  int, number of communication rounds that the federated clients average results for
    :param clients:                 int_array, the maximum number of clients participating in a communication round
    :return:
    """

    # Load data
    train_data, train_labels, test_data, test_labels, dataset = Data_Loader.load_data(dataset)

    # Perform Experiments
    for client_num in clients:
        experiment_federated(client_num, dataset, experiment, train_data, train_labels, test_data, test_labels, rounds,
                             split='overlap', participants=10)

    experiment_centralized(dataset, experiment, train_data, train_labels, test_data, test_labels, rounds)


# ------------------------------------------------ End Experiments - 2 --------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# -------------------------------------------------- Experiments - 3 ----------------------------------------------- #

def experiment_6_differential_federated_learning(dataset, experiment, rounds, clients, sigmas):
    # Load data
    train_data, train_labels, test_data, test_labels, dataset = Data_Loader.load_data(dataset)

    # Perform Experiments
    for sigma in sigmas:
        experiment_differential_privacy(clients, dataset, experiment, train_data, train_labels, test_data,
                                        test_labels, sigma=sigma, rounds=rounds, split='overlap', participants=50,
                                        learning_rate=0.01)

    experiment_centralized(dataset, experiment, train_data, train_labels, test_data, test_labels, rounds)


# ------------------------------------------------ End Experiments - 3 --------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


if __name__ == '__main__':
    # Experiment 1 - Number of clients
    # num_clients = [10]
    # experiment_1_number_of_clients(dataset="MNIST", experiment="CLIENTS", rounds=10, clients=num_clients)
    # Experiment 4 - Number of clients no overlap
    # num_clients = [10]
    # experiment_4_split_digits(dataset="MNIST", experiment="SPLIT_DIGITS", rounds=100, clients=num_clients)
    # plot_results(dataset="MNIST", experiment="SPLIT_DIGITS", keys=num_clients, date="2019-07-10",
    #              suffix=str(num_clients), move=True)

    # Experiment 5
    # num_clients = [100]
    # experiment_5_split_digits_with_overlap(dataset="MNIST", experiment="SPLIT_DIGITS_OVERLAP", rounds=300,
    #                                        clients=num_clients)
    # plot_results(dataset="MNIST", experiment="SPLIT_DIGITS_OVERLAP", keys=num_clients, date="2019-07-10",
    #              suffix=str(num_clients), move=True)

    sigma_arr = [1]
    experiment_6_differential_federated_learning(dataset="MNIST", experiment="DIFF_PRIV", rounds=200, clients=100,
                                                 sigmas=sigma_arr)
