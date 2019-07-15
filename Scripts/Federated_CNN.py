import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

import Scripts.Data_Loader_Functions as Data_Loader
from Scripts import Centralized_CNN as cNN
from Scripts import Model_Reset as Reset
from Scripts import Print_Functions as Output
from Scripts.Weights_Accountant import WeightsAccountant

models = tf.keras.models  # like 'from tensorflow.keras import models' (PyCharm import issue workaround)

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------ Paths ----------------------------------------------------- #

FEDERATED_GLOBAL_MODEL = os.path.join(cNN.MODELS, "federated_global_model.json")
FEDERATED_GLOBAL_WEIGHTS = os.path.join(cNN.MODELS, "federated_global_weights.npy")
FEDERATED_LOCAL_WEIGHTS_PATH = os.path.join(cNN.MODELS, "Federated Weights")
FEDERATED_LOCAL_WEIGHTS = os.path.join(FEDERATED_LOCAL_WEIGHTS_PATH, "federated_local_weights_client_{}")


# ---------------------------------------------------- End Paths --------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Utility Functions ----------------------------------------------- #


def reset_federated_model():
    """
    Deletes the global model and weights, as well as all local weights (if any)
    :return:
    """
    if os.path.isdir(FEDERATED_LOCAL_WEIGHTS_PATH):
        Reset.remove_files(FEDERATED_LOCAL_WEIGHTS_PATH)
    else:
        os.mkdir(FEDERATED_LOCAL_WEIGHTS_PATH)
    if os.path.isdir(cNN.MODELS):
        Reset.remove_files(cNN.MODELS)
    else:
        os.mkdir(cNN.MODELS)


def split_data_into_clients(num_of_clients, train_data, train_labels):
    """
    Splits a dataset into a provided number of clients to simulate a "federated" setting

    :param num_of_clients:          integer specifying the number of clients the data should be split into
    :param train_data:              numpy array
    :param train_labels:            numpy array

    :return:
        train_data:                 numpy array (with additional dimension for N clients)
        train_labels:               numpy array (with additional dimension for N clients)
    """

    # Split data into twice as many shards as clients
    train_data = np.array_split(train_data, num_of_clients * 2)
    train_labels = np.array_split(train_labels, num_of_clients * 2)

    # Shuffle shards so that for sorted data, shards with different labels are adjacent
    train = list(zip(train_data, train_labels))
    np.random.shuffle(train)
    train_data, train_labels = zip(*train)

    # Concatenate adjacent shards
    train_data = [np.concatenate(train_data[i:i+2]) for i in range(0, len(train_data), 2)]
    train_labels = [np.concatenate(train_labels[i:i+2]) for i in range(0, len(train_labels), 2)]

    return train_data, train_labels


def create_client_index_array(num_of_clients, num_participating_clients=None):
    """
    Creates a random integer array used to select clients for a communication round, e.g. clients [2, 0, 7, 5].
    If no number of participating clients for a given round is specified, all clients participate in a communication
    round. E.g. if num_of_clients = 5, then num_participating_clients [0, 1, 2, 3, 4].

    :param num_of_clients:              int, specifying the total number of clients available
    :param num_participating_clients:   int, specifying how many clients participate in a given communication round

    :return:
        clients:                        numpy array
    """

    if num_participating_clients:
        clients = np.random.choice(num_of_clients, num_participating_clients)
    else:
        clients = np.arange(num_of_clients)

    return clients


def init_global_model(input_shape=(28, 28, 1)):
    """
    Initializes a global "server-side" model.
    :param input_shape:              tuple, input shape of one training example (default, MNIST shape)

    :return:
        model                       tensorflow-graph
    """

    # Build the model
    model = cNN.build_cnn(input_shape=input_shape)

    # Compile the model
    model.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Save initial model
    json_config = model.to_json()
    with open(FEDERATED_GLOBAL_MODEL, 'w') as json_file:
        json_file.write(json_config)

    return model


# ---------------------------------------------- End Utility Functions --------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Federated Learning ---------------------------------------------- #


def train_client_model(client, epochs, model, train_data, train_labels, weights_accountant):
    model = cNN.train_cnn(model, train_data[client], train_labels[client], epochs=epochs)
    weights = model.get_weights()
    weights_accountant.append_local_weights(weights)


def client_learning(model, client, epochs, train_data, train_labels, weights_accountant):
    Output.print_client_id(client)

    # Initialize model structure and load weights
    weights = weights_accountant.get_global_weights()
    model.set_weights(weights)

    # Train local model and store weights to folder
    train_client_model(client, epochs, model, train_data, train_labels, weights_accountant)


def communication_round(model, num_of_clients, train_data, train_labels, epochs, weights_accountant,
                        num_participating_clients=None):
    """
    One round of communication between a 'server' and the 'clients'. Each client 'downloads' a global model and trains
    a local model, updating its weights locally. When all clients have updated their weights, they are 'uploaded' to
    the server and averaged.

    :param model:                           Tensorflow Graph
    :param num_of_clients:                  int, number of clients globally available
    :param train_data:                      numpy array
    :param train_labels:                    numpy array
    :param epochs:                          int, number of epochs each client will train in a given communication round
    :param weights_accountant:              WeightsAccountant object
    :param num_participating_clients:       int, number of participating clients in a given communication round
    :return:
    """

    # Select clients to participate in communication round
    clients = create_client_index_array(num_of_clients, num_participating_clients)

    # Train each client
    for client in clients:
        client_learning(model, client, epochs, train_data, train_labels, weights_accountant)

    # Average all local updates and store them as new 'global weights'
    weights_accountant.average_local_weights()


def federated_learning(communication_rounds, num_of_clients, train_data, train_labels, test_data, test_labels,
                       epochs, num_participating_clients=None):
    """
    Train a federated model for a specified number of rounds until convergence.

    :param communication_rounds:            int, number of times the global weights will be updated
    :param num_of_clients:                  int, number of clients globally available
    :param train_data:                      numpy array
    :param train_labels:                    numpy array
    :param test_data:                       numpy array
    :param test_labels:                     numpy array
    :param epochs:                          int, number of epochs each client will train in a given communication round
    :param num_participating_clients:       int, number of participating clients in a given communication round

    :return:
        history                             pandas data-frame, contains the history of loss & accuracy values off all
                                            communication rounds
    """

    # Create empty data-frame
    history = pd.DataFrame(columns=['Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy'])

    # Initialize a random global model and store the weights
    model = init_global_model()
    weights = model.get_weights()

    clients = num_participating_clients if num_participating_clients is not None else num_of_clients
    weights_accountant = WeightsAccountant(weights, clients=clients)
    # Start communication rounds and save the results of each round to the data frame
    for _ in range(communication_rounds):
        Output.print_communication_round(_ + 1)
        communication_round(model, num_of_clients, train_data, train_labels, epochs, weights_accountant,
                            num_participating_clients)
        test_loss, test_acc = evaluate_federated_cnn(test_data, test_labels, model, weights_accountant)

        history = history.append(pd.Series([test_loss, test_acc],
                                           index=['Test Loss', 'Test Accuracy']),
                                 ignore_index=True)
    weights = weights_accountant.get_global_weights()
    np.save(FEDERATED_GLOBAL_WEIGHTS, weights)
    return history


def evaluate_federated_cnn(test_data, test_labels, model=None, weights_accountant=None, train_data=None,
                           train_labels=None):
    """
    Evaluate the global CNN.

    :param test_data:                       numpy array
    :param test_labels:                     numpy array
    :param model:                           Tensorflow graph
    :param weights_accountant:              WeightsAccountant object
    :param train_labels:                    numpy array, optional
    :param train_data:                      numpy array, optional

    :return:
        test_loss                           float
        test_acc                            float
    """

    if not model:
        model = init_global_model()
    if weights_accountant:
        weights = weights_accountant.get_global_weights()
    else:
        weights = np.load(FEDERATED_GLOBAL_WEIGHTS, allow_pickle=True)
    model.set_weights(weights)

    results = cNN.evaluate_cnn(model, test_data, test_labels, train_data, train_labels)

    if train_data is not None and train_labels is not None:
        test_loss, test_acc, train_loss, train_acc = results[0], results[1], results[2], results[3]

        Output.print_loss_accuracy(train_acc, train_loss, "Train")
        Output.print_loss_accuracy(test_acc, test_loss, "Test")
        return test_loss, test_acc, train_loss, train_acc

    else:
        test_loss, test_acc = results[0], results[1]
        Output.print_loss_accuracy(test_acc, test_loss, "Test")
        return test_loss, test_acc


# ---------------------------------------------- End Federated Learning -------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


def main(clients, rounds=30, participants=5, dataset="MNIST", training=True, evaluating=True, plotting=False,
         max_samples=None):
    """
    Main function including a number of flags that can be set

    :param clients:             int, specifying number of participating clients
    :param rounds:              int, number of communication rounds
    :param participants:        int, number of clients participating in a given round
    :param dataset:             string (selecting the data set to be used, default is MNIST)
    :param training:            bool
    :param plotting:            bool
    :param evaluating:          bool
    :param max_samples:         int

    :return:
    """

    # Load data
    train_data, train_labels, test_data, test_labels, dataset = Data_Loader.load_data(dataset)

    if max_samples:
        train_data = train_data[:max_samples]
        train_labels = train_labels[:max_samples]

    # Split training data
    train_data, train_labels = split_data_into_clients(clients, train_data, train_labels)

    if training:
        reset_federated_model()

        # Train Model
        history = federated_learning(communication_rounds=rounds,
                                     num_of_clients=clients,
                                     train_data=train_data,
                                     train_labels=train_labels,
                                     test_data=test_data,
                                     test_labels=test_labels,
                                     epochs=5,
                                     num_participating_clients=participants)

        # Save history for plotting
        file = time.strftime("%Y-%m-%d-%H%M%S") + r"_{}_rounds_{}_clients_{}.csv".format(dataset, rounds, clients)
        history.to_csv(os.path.join(cNN.RESULTS, file))

    # Evaluate model
    if evaluating:
        evaluate_federated_cnn(test_data, test_labels)

    # Plot Accuracy and Loss
    if plotting:
        # Open most recent history file
        files = os.listdir(cNN.RESULTS)
        files = [os.path.join(cNN.RESULTS, file) for file in files]
        latest_file = max(files, key=os.path.getctime)
        history = pd.read_csv(latest_file)

        # Plot the data
        Output.plot_federated_accuracy(history)
        Output.plot_federated_loss(history)
        # Output.display_images(train_data, train_labels)


if __name__ == '__main__':
    main(clients=100, rounds=30, training=True, plotting=False, evaluating=True, participants=10)
