import os
import time

import pandas as pd

import Scripts.Centralized_CNN as cNN
import Scripts.Federated_CNN as fed_CNN
import Scripts.Print_Functions as Output


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
    file = time.strftime("%Y-%m-%d-%H%M%S") + r"_Combined_{}_rounds_{}_experiment_{}.csv".format(dataset, rounds,
                                                                                                 experiment)
    history.to_csv(os.path.join(cNN.RESULTS, file))
    return history


def plot_results(dataset, rounds, experiment, keys):
    history = combine_results(dataset, experiment, rounds, keys)

    # Plot Accuracy
    params = Output.PlotParams(
        metric='Accuracy',
        title='Model Accuracy',
        x_label='Federated Comm. Round/Centralized Epoch',
        y_label='Accuracy',
        legend_loc='upper left',
        num_format="{:5.1%}"
    )
    Output.plot_joint_metric(history, params)

    # Plot Loss
    params = Output.PlotParams(
        metric='Loss',
        title='Model Loss',
        x_label='Federated Comm. Round/Centralized Epoch',
        y_label='Loss',
        legend_loc='bottom left',
        num_format="{:5.2f}"
    )
    Output.plot_joint_metric(history, params)


# ---------------------------------------------- End Utility Functions --------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Experiment Runners ---------------------------------------------- #


def experiment_centralized(dataset, train_data, train_labels, test_data, test_labels, epochs=5):
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
    file = time.strftime("%Y-%m-%d-%H%M%S") + r"_Centralized_{}.csv".format(dataset)
    history.to_csv(os.path.join(cNN.RESULTS, file))


def experiment_federated(clients, dataset, train_data, train_labels, test_data, test_labels,
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
    file = time.strftime("%Y-%m-%d-%H%M%S") + r"_Federated_{}_rounds_{}_clients_{}.csv".format(dataset, rounds, clients)
    history = history.rename(index=str, columns={"Train Loss": "Federated Train Loss",
                                                 "Train Accuracy": "Federated Train Accuracy",
                                                 "Test Loss": "Federated Test Loss",
                                                 "Test Accuracy": "Federated Test Accuracy"})
    history.to_csv(os.path.join(cNN.RESULTS, file))


# ---------------------------------------------- End Experiment Runners -------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# ---------------------------------------------------- Experiments ------------------------------------------------- #


def experiment_1_number_of_clients(dataset, rounds, clients):
    # Load data
    train_data, train_labels, test_data, test_labels, dataset = cNN.load_data(dataset)

    # Perform Experiments
    experiment_centralized(dataset, train_data, train_labels, test_data, test_labels,
                           rounds)

    for client_num in clients:
        experiment_federated(client_num, dataset, train_data, train_labels, test_data,
                             test_labels,
                             rounds)

    # Plot results
    plot_results(dataset, rounds=rounds, experiment="CLIENTS", keys=clients)


# -------------------------------------------------- End Experiments ----------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


if __name__ == '__main__':

    # Experiment 1 - Number of clients
    num_clients = [2, 5, 10, 20, 50, 100]
    experiment_1_number_of_clients(dataset="MNIST", rounds=5, clients=num_clients)
