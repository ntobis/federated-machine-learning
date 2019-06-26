import os
import sys
import time

import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Scripts import Centralized_CNN as cNN
from Scripts import Federated_CNN as fed_CNN
from Scripts import Print_Functions as Output

pd.set_option('display.max_columns', 500)

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Utility Functions ----------------------------------------------- #


def combine_results(dataset, experiment, rounds, keys):
    # Open most recent history files
    files = os.listdir(cNN.RESULTS)
    files = [os.path.join(cNN.RESULTS, file) for file in files]

    # Combine Results
    sorted_files = sorted(files, key=os.path.getctime)
    centralized = pd.read_csv(sorted_files[-1], index_col=0)
    history = centralized

    for idx in range(len(keys)):
        federated = pd.read_csv(sorted_files[-idx - 2], index_col=0)
        federated = federated.rename(index=str,
                                     columns={"Federated Train Loss": "Federated Train Loss {} {}".format(
                                         experiment,
                                         keys[-idx - 1]),
                                         "Federated Train Accuracy": "Federated Train Accuracy {} {}".format(
                                             experiment,
                                             keys[-idx - 1]),
                                         "Federated Test Loss": "Federated Test Loss {} {}".format(
                                             experiment,
                                             keys[-idx - 1]),
                                         "Federated Test Accuracy": "Federated Test Accuracy {} {}".format(
                                             experiment,
                                             keys[-idx - 1])})
        history = pd.concat([history.reset_index(drop=True), federated.reset_index(drop=True)], axis=1)

    # Store combined results
    # file = time.strftime("%Y-%m-%d-%H%M%S") + r"_Combined_{}_rounds_{}_experiment_{}.csv".format(dataset, rounds,
    #                                                                                              experiment)
    # history.to_csv(os.path.join(cNN.RESULTS, file))
    return history


def plot_results(dataset, rounds, experiment, keys):
    history = combine_results(dataset, experiment, rounds, keys)

    # Plot Accuracy
    params = Output.PlotParams(
        dataset=dataset,
        experiment=experiment,
        metric='Accuracy',
        title='Model Accuracy',
        x_label='Federated Comm. Round/Centralized Epoch',
        y_label='Accuracy',
        legend_loc='upper left',
        num_format="{:5.1%}",
        max_epochs=None
    )
    Output.plot_joint_metric(history, params)

    # Plot Loss
    params = Output.PlotParams(
        dataset=dataset,
        experiment=experiment,
        metric='Loss',
        title='Model Loss',
        x_label='Federated Comm. Round/Centralized Epoch',
        y_label='Loss',
        legend_loc='lower left',
        num_format="{:5.2f}",
        max_epochs=None
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
# ---------------------------------------------------- Experiments ------------------------------------------------- #


def experiment_1_number_of_clients(dataset, experiment, rounds, clients):
    # Load data
    train_data, train_labels, test_data, test_labels, dataset = cNN.load_data(dataset)

    # Perform Experiments
    for client_num in clients:
        experiment_federated(client_num, dataset, experiment, train_data, train_labels, test_data, test_labels, rounds)

    experiment_centralized(dataset, experiment, train_data, train_labels, test_data, test_labels, rounds)

    # Plot results
    plot_results(dataset, rounds=rounds, experiment=experiment, keys=clients)


def experiment_2_limited_digits(dataset, experiment, rounds, digit_array):
    # Load data
    train_data, train_labels, test_data, test_labels, dataset = cNN.load_data(dataset)

    # Perform Experiments
    for digits in digit_array:
        train_data_filtered = train_data[np.in1d(train_labels, digits)]
        train_labels_filtered = train_labels[np.in1d(train_labels, digits)]
        test_data_filtered = test_data[np.in1d(test_labels, digits)]
        test_labels_filtered = test_labels[np.in1d(test_labels, digits)]

        experiment_federated(10, dataset, experiment, train_data_filtered, train_labels_filtered, test_data_filtered,
                             test_labels_filtered, rounds)

        experiment_centralized(dataset, experiment, train_data_filtered, train_labels_filtered, test_data_filtered,
                               test_labels_filtered, rounds)

        # Plot results
        plot_results(dataset, rounds=rounds, experiment=experiment + str(digits), keys=[digits])


def experiment_3_add_noise(dataset, experiment, rounds, std_devs):
    # Load data
    train_data, train_labels, test_data, test_labels, dataset = cNN.load_data(dataset)

    # Perform Experiments
    for std_dev in std_devs:
        experiment = experiment + "_" + str(std_dev)
        train_data_noise = train_data + np.random.normal(loc=0, scale=std_dev, size=train_data.shape)
        experiment_federated(10, dataset, experiment, train_data_noise, train_labels, test_data, test_labels, rounds)

        experiment_centralized(dataset, experiment, train_data_noise, train_labels, test_data, test_labels, rounds)

        # Plot results
        plot_results(dataset, rounds=rounds, experiment=experiment, keys=[std_devs])


# -------------------------------------------------- End Experiments ----------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


if __name__ == '__main__':
    # Experiment 1 - Number of clients
    num_clients = [2, 5, 10, 20, 50, 100]
    experiment_1_number_of_clients(dataset="MNIST", experiment="CLIENTS", rounds=30, clients=num_clients)

    # Experiment 2 - Digits
    digits_arr = [[0, 5], [0, 2, 5, 9], [0, 2, 4, 5, 7, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    experiment_2_limited_digits(dataset="MNIST", experiment="DIGITS", rounds=30, digit_array=digits_arr)

    # Experiment 3 - Adding Noise
    std_dev_arr = [0.1, 0.25, 0.5]
    experiment_3_add_noise(dataset="MNIST", experiment="NOISE", rounds=30, std_devs=std_dev_arr)
