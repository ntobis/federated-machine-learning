import pandas as pd
import numpy as np

from Scripts._old import Federated_CNN as fedCNN
from Scripts import Print_Functions as Output
from Scripts._old.Federated_CNN import FEDERATED_GLOBAL_WEIGHTS
from Scripts.Weights_Accountant import WeightsAccountant


def communication_round(model, num_of_clients, train_data, train_labels, epochs, weights_accountant,
                        sigma=None, num_participating_clients=None):
    """
    Differentially private round of communication between a 'server' and the 'clients'. Each client 'downloads' a global
    model and trains a local model, updating its weights locally. Different from the naive averaging in federated
    learning, the weights accountant adds noise to the updates of the weights in order to ensure client-level
    differential privacy. When all clients have updated their weights, they are 'uploaded' to the server and averaged.

    :param model:                           Tensorflow Graph
    :param num_of_clients:                  int, number of clients globally available
    :param train_data:                      numpy array
    :param train_labels:                    numpy array
    :param epochs:                          int, number of epochs each client will train in a given communication round
    :param weights_accountant:              WeightsAccountant object
    :param sigma:                           float, determining the level of differential privacy
    :param num_participating_clients:       int, number of participating clients in a given communication round
    :return:
    """

    # Select clients to participate in communication round
    clients = fedCNN.create_client_index_array(num_of_clients, num_participating_clients)

    # Train each client
    for client in clients:
        fedCNN.client_learning(model, client, epochs, train_data, train_labels, weights_accountant)

    # Update Weights with a Gaussian Mechanism
    weights_accountant.compute_local_updates()
    weights_accountant.compute_update_norm()
    weights_accountant.clip_updates()
    weights_accountant.average_local_clipped_updates()
    weights_accountant.compute_noisy_global_weights(sigma)


def federated_learning(communication_rounds, num_of_clients, train_data, train_labels, test_data, test_labels,
                       epochs, sigma, num_participating_clients=None, learning_rate=0.01):
    """
    Train a federated model for a specified number of rounds until convergence.

    :param communication_rounds:            int, number of times the global weights will be updated
    :param num_of_clients:                  int, number of clients globally available
    :param train_data:                      numpy array
    :param train_labels:                    numpy array
    :param test_data:                       numpy array
    :param test_labels:                     numpy array
    :param epochs:                          int, number of epochs each client will train in a given communication round
    :param sigma:                           float, sigma defining differential privacy level
    :param num_participating_clients:       int, number of participating clients in a given communication round
    :param learning_rate:                   float, specifying the learning rate of the local algorithms

    :return:
        history                             pandas data-frame, contains the history of loss & accuracy values off all
                                            communication rounds
    """

    # Create empty data-frame
    history = pd.DataFrame(columns=['Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy'])

    # Initialize a random global model and store the weights
    model = fedCNN.init_global_model(learning_rate=learning_rate)
    weights = model.get_weights()
    clients = num_participating_clients if num_participating_clients is not None else num_of_clients
    weights_accountant = WeightsAccountant(weights)
    # Start communication rounds and save the results of each round to the data frame
    for _ in range(communication_rounds):
        Output.print_communication_round(_ + 1)
        communication_round(model, num_of_clients, train_data, train_labels, epochs, weights_accountant,
                            sigma, num_participating_clients)
        test_loss, test_acc = fedCNN.evaluate_federated_cnn(test_data, test_labels, model, weights_accountant)

        history = history.append(pd.Series([test_loss, test_acc],
                                           index=['Test Loss', 'Test Accuracy']),
                                 ignore_index=True)
    weights = weights_accountant.get_global_weights()
    np.save(FEDERATED_GLOBAL_WEIGHTS, weights)
    return history
