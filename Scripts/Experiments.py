import sys
import os

import Scripts.Data_Loader_Functions as Data_Loader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time

import pandas as pd
import numpy as np

from Scripts import Centralized_CNN as cNN
from Scripts import Federated_CNN as fed_CNN
from Scripts import Print_Functions as Output

pd.set_option('display.max_columns', 500)

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Utility Functions ----------------------------------------------- #


def combine_results(experiment, keys, sub_folder=None):
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


def plot_results(dataset, experiment, keys, date, suffix):
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


# ---------------------------------------------- End Utility Functions --------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Experiment Runners ---------------------------------------------- #


def experiment_centralized(dataset, experiment, train_data, train_labels, test_data, test_labels, epochs=5):
    # Train Centralized CNN
    centralized_model = cNN.build_cnn(input_shape=(28, 28, 1))
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
                         rounds=5, epochs=1):
    # Split data
    if dataset is not "AUTISM_BODY":
        train_data, train_labels = fed_CNN.split_data_into_clients(clients, train_data, train_labels)

    # Train federated model
    fed_CNN.reset_federated_model()

    # Build initial model
    model = cNN.build_cnn(input_shape=(28, 28, 1))

    # Save initial model
    json_config = model.to_json()
    with open(fed_CNN.FEDERATED_GLOBAL_MODEL, 'w') as json_file:
        json_file.write(json_config)

    # Train Model
    history = fed_CNN.federated_learning(communication_rounds=rounds,
                                         num_of_clients=clients,
                                         train_data=train_data,
                                         train_labels=train_labels,
                                         test_data=test_data,
                                         epochs=epochs,
                                         test_labels=test_labels)

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
    # Load data
    train_data, train_labels, test_data, test_labels, dataset = Data_Loader.load_data(dataset)

    # Perform Experiments
    for client_num in clients:
        experiment_federated(client_num, dataset, experiment, train_data, train_labels, test_data, test_labels, rounds)

    experiment_centralized(dataset, experiment, train_data, train_labels, test_data, test_labels, rounds)


def experiment_2_limited_digits(dataset, experiment, rounds, digit_array):
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


def experiment_main_1():
    # Experiment 1 - Number of clients
    num_clients = [2, 5, 10, 20, 50, 100]
    experiment_1_number_of_clients(dataset="MNIST", experiment="CLIENTS", rounds=30, clients=num_clients)

    # Plot results
    plot_results(dataset="MNIST", experiment="CLIENTS", keys=num_clients, date="2019-06-25",
                 suffix=str(num_clients))

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


# ------------------------------------------------ End Experiments - 1 --------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# -------------------------------------------------- Experiments - 2 ----------------------------------------------- #

def experiment_4_autism():
    features, labels = Data_Loader.load_autism_data_body()
    train_data_clients, train_labels_clients, test_data_clients, test_labels_clients = [], [], [], []
    for idx, client in enumerate(features):
        train_data, train_labels, test_data, test_labels = Data_Loader.train_test_split(features[idx], labels[idx])
        train_data_clients.append(train_data)
        train_labels_clients.append(train_labels)
        test_data_clients.append(test_data)
        test_labels_clients.append(test_labels)

    experiment_federated(len(train_data_clients), "AUTISM_BODY", "TEST", train_data_clients, train_labels_clients, test_data_clients, test_labels_clients)


# ------------------------------------------------ End Experiments - 2 --------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


if __name__ == '__main__':
    print("AUTISM")
    experiment_4_autism()
    # Experiment 1 - Number of clients
    num_clients = [2]
    print("MNIST")
    experiment_1_number_of_clients(dataset="MNIST", experiment="CLIENTS", rounds=30, clients=num_clients)