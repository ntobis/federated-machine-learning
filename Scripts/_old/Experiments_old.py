import os
import time

import numpy as np
import pandas as pd

from Scripts import Print_Functions as Output, Data_Loader_Functions as dL
from Scripts.Experiments import RESULTS
from Scripts._old import Centralized_CNN as cNN, Federated_CNN as fed_CNN


def runner_centralized_mnist(dataset, experiment, train_data, train_labels, test_data, test_labels, epochs=5):
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
    centralized_model = cNN.build_cnn(input_shape=train_data[0].shape)
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


def runner_federated_mnist(clients, dataset, experiment, train_data, train_labels, test_data, test_labels,
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

    train_data, train_labels = dL.split_data_into_clients(clients, split, train_data, train_labels)

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
    train_data, train_labels, test_data, test_labels, dataset = dL.load_data(dataset)

    # Perform Experiments
    for client_num in clients:
        runner_federated_mnist(client_num, dataset, experiment, train_data, train_labels, test_data, test_labels,
                               rounds)

    runner_centralized_mnist(dataset, experiment, train_data, train_labels, test_data, test_labels, rounds)


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
    train_data, train_labels, test_data, test_labels, dataset = dL.load_data(dataset)

    # Perform Experiments
    for digit in digit_array:
        experiment = experiment + "_" + str(digit)
        train_data_filtered = train_data[np.in1d(train_labels, digit)]
        train_labels_filtered = train_labels[np.in1d(train_labels, digit)]
        test_data_filtered = test_data[np.in1d(test_labels, digit)]
        test_labels_filtered = test_labels[np.in1d(test_labels, digit)]

        runner_federated_mnist(10, dataset, experiment, train_data_filtered, train_labels_filtered, test_data_filtered,
                               test_labels_filtered, rounds)

        runner_centralized_mnist(dataset, experiment, train_data_filtered, train_labels_filtered, test_data_filtered,
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
    train_data, train_labels, test_data, test_labels, dataset = dL.load_data(dataset)

    # Perform Experiments
    for std_dv in std_devs:
        this_experiment = experiment + "_" + str(std_dv)
        train_data_noise = train_data + np.random.normal(loc=0, scale=std_dv, size=train_data.shape)
        Output.display_images(train_data_noise, train_labels)
        runner_federated_mnist(10, dataset, this_experiment, train_data_noise, train_labels, test_data, test_labels,
                               rounds)

        runner_centralized_mnist(dataset, this_experiment, train_data_noise, train_labels, test_data, test_labels,
                                 rounds)


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
    train_data, train_labels, test_data, test_labels, dataset = dL.load_data(dataset)

    # Perform Experiments
    for client_num in clients:
        runner_federated_mnist(client_num, dataset, experiment, train_data, train_labels, test_data, test_labels,
                               rounds,
                               split='no_overlap')

    runner_centralized_mnist(dataset, experiment, train_data, train_labels, test_data, test_labels, rounds)


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
    train_data, train_labels, test_data, test_labels, dataset = dL.load_data(dataset)

    # Perform Experiments
    for client_num in clients:
        runner_federated_mnist(client_num, dataset, experiment, train_data, train_labels, test_data, test_labels,
                               rounds,
                               split='overlap', participants=10)

    runner_centralized_mnist(dataset, experiment, train_data, train_labels, test_data, test_labels, rounds)


def move_results(experiment, date, keys):
    """
    Utility function used to move results into a specified folder with a name of the format:
    EXPERIMENT DATE KEYS

    :param experiment:                  string, name of the experiment
    :param date:                        string, date of the experiment
    :param keys:                        list, type of experiment, e.g. number of clients: [2, 5, 10, 100]
    :return:
    """

    experiment_path = os.path.join(RESULTS, experiment + " " + date + " " + str(keys))
    if not os.path.isdir(experiment_path):
        os.mkdir(experiment_path)
    for elem in os.listdir(RESULTS):
        if os.path.isfile(os.path.join(RESULTS, elem)):
            old_loc = os.path.join(RESULTS, elem)
            new_loc = os.path.join(RESULTS, experiment_path, elem)
            os.rename(old_loc, new_loc)


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
        files = os.listdir(os.path.join(RESULTS, sub_folder))
        files = [os.path.join(os.path.join(RESULTS, sub_folder), file) for file in files]
    else:
        files = os.listdir(RESULTS)
        files = [os.path.join(RESULTS, file) for file in files]

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