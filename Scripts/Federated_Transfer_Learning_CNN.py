import os

import numpy as np
import tensorflow as tf

from Scripts import Centralized_CNN as cNN
from Scripts import Centralized_Pain_CNN as painCNN
from Scripts import Model_Reset as Reset
from Scripts import Print_Functions as Output
from Scripts.Weights_Accountant import WeightsAccountant

models = tf.keras.models  # like 'from tensorflow.keras import models' (PyCharm import issue workaround)
# optimizer = tf.keras.optimizers

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------ Paths ----------------------------------------------------- #

FEDERATED_GLOBAL_MODEL = os.path.join(cNN.MODELS, 'Pain', 'Federated', 'Global Model', "federated_global_model.json")
FEDERATED_GLOBAL_WEIGHTS = os.path.join(cNN.MODELS, 'Pain', 'Federated', 'Global Model', "federated_global_weights.npy")
FEDERATED_LOCAL_WEIGHTS_PATH = os.path.join(cNN.MODELS, 'Pain', 'Federated', 'Federated Weights')
FEDERATED_LOCAL_WEIGHTS = os.path.join(FEDERATED_LOCAL_WEIGHTS_PATH, "federated_local_weights_client_{}")


# ---------------------------------------------------- End Paths --------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

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


def init_global_model(optimizer, loss, metrics, input_shape=(215, 215, 1)):
    """
    Initializes a global "server-side" model.
    :param metrics:
    :param loss:
    :param optimizer:
    :param input_shape:             tuple, input shape of one training example (default, MNIST shape)

    :return:
        model                       tensorflow-graph
    """

    # Build the model
    model = painCNN.build_cnn(input_shape=input_shape)

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Save initial model
    json_config = model.to_json()
    with open(FEDERATED_GLOBAL_MODEL, 'w') as json_file:
        json_file.write(json_config)

    return model


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Federated Learning ---------------------------------------------- #


def train_client_model(client, epochs, model, train_data=None, train_labels=None, df=None, weights_accountant=None,
                       session=None):
    """
    Utility function training a simple CNN for 1 client in a federated setting and adding those weights to the
    weights_accountant. Call this function in a federated loop that then makes the weights_accountant average the
    weights to send to a global model.

    :param df:
    :param session:
    :param client:                      int, index for a specific client to be trained
    :param epochs:                      int, local epochs to be trained
    :param model:                       Tensorflow Graph
    :param train_data:                  numpy array, partitioned into a number of clients
    :param train_labels:                numpy array, partitioned into a number of clients
    :param weights_accountant:          WeightsAccountant object
    :return:
    """

    old_weights = model.get_weights()
    if train_data is not None and train_labels is not None:
        model, history = painCNN.train_cnn(model, epochs, train_data[client], train_labels[client], evaluate=False)
    elif df is not None:
        model, history = painCNN.train_cnn(model, epochs, df=df, evaluate=False, session=session)
    else:
        raise ValueError('Need to provide either "train_data" and "train_labels", or "df", None was provided.')
    weights = model.get_weights()

    # Only append weights for updating the model, if there was an update
    if not all([np.array_equal(w_1, w_2) for w_1, w_2 in zip(old_weights, weights)]):
        weights_accountant.append_local_weights(weights)


def client_learning(model, client, epochs, train_data=None, train_labels=None, df=None, weights_accountant=None,
                    session=None):
    """
    Initializes a client model and kicks off the training of that client by calling "train_client_model".

    :param session:
    :param df:
    :param model:                       Tensorflow graph
    :param client:                      int, index for a specific client to be trained, or array tb converted to int
    :param epochs:                      int, local epochs to be trained
    :param train_data:                  numpy array, partitioned into a number of clients
    :param train_labels:                numpy array, partitioned into a number of clients
    :param weights_accountant:          WeightsAccountant object
    :return:
    """

    # Initialize model structure and load weights
    weights = weights_accountant.get_global_weights()
    model.set_weights(weights)

    # Train local model and store weights to folder
    train_client_model(client, epochs, model, train_data, train_labels, df, weights_accountant, session=session)


def communication_round(model, clients, train_data=None, train_labels=None, df=None, epochs=1, weights_accountant=None,
                        num_participating_clients=None, session=None):
    """
    One round of communication between a 'server' and the 'clients'. Each client 'downloads' a global model and trains
    a local model, updating its weights locally. When all clients have updated their weights, they are 'uploaded' to
    the server and averaged.

    :param session:
    :param df:
    :param model:                           Tensorflow Graph
    :param clients:                         int, number of clients globally available, or array
    :param train_data:                      numpy array
    :param train_labels:                    numpy array
    :param epochs:                          int, number of epochs each client will train in a given communication round
    :param weights_accountant:              WeightsAccountant object
    :param num_participating_clients:       int, number of participating clients in a given communication round
    :return:
    """

    # Select clients to participate in communication round
    if type(num_participating_clients) is int:
        clients = create_client_index_array(clients, num_participating_clients)
    if clients is None and df is not None:
        clients = df['Person'].unique()

    # Train each client
    for idx, client in enumerate(clients):
        df_train = df[df['Person'] == client]
        Output.print_client_id(client[0, 0].astype(int)) if not np.issubdtype(client, np.integer) else \
            Output.print_client_id(client)
        client_learning(model, idx, epochs, train_data, train_labels, df_train, weights_accountant, session)

    # Average all local updates and store them as new 'global weights'
    if weights_accountant is not None:
        weights_accountant.average_local_weights()


def federated_learning(communication_rounds, num_of_clients, train_data, train_labels, test_data, test_labels, epochs,
                       num_participating_clients=None, people=None, model=None,
                       optimizer=None, loss=None, metrics=None, session=False, df=None, evaluate=True):
    """
    Train a federated model for a specified number of rounds until convergence.

    :param evaluate:
    :param df:
    :param session:
    :param metrics:
    :param loss:
    :param optimizer:
    :param model:
    :param communication_rounds:            int, number of times the global weights will be updated
    :param num_of_clients:                  int, number of clients globally available
    :param train_data:                      numpy array
    :param train_labels:                    numpy array
    :param test_data:                       numpy array
    :param test_labels:                     numpy array
    :param epochs:                          int, number of epochs each client will train in a given communication round
    :param num_participating_clients:       int, number of participating clients in a given communication round
    :param people:                          numpy array, of length test_labels, used to enable individual metric logging

    :return:
        history                             pandas data-frame, contains the history of loss & accuracy values off all
                                            communication rounds
    """

    # Create history object
    history = painCNN.history_set_up(people)

    # Initialize a random global model and store the weights
    if model is None:
        model = init_global_model(optimizer, loss, metrics)
    weights = model.get_weights()

    clients = num_participating_clients if num_participating_clients is not None else num_of_clients
    weights_accountant = WeightsAccountant(weights, clients=clients)

    # Start communication rounds and save the results of each round to the data frame
    for comm_round in range(communication_rounds):
        Output.print_communication_round(comm_round + 1)
        communication_round(model, num_of_clients, train_data, train_labels, df, epochs, weights_accountant,
                            num_participating_clients, session)
        if evaluate:
            history = evaluate_federated_cnn(comm_round, test_data, test_labels, df, model, weights_accountant, history,
                                             people, optimizer, loss, metrics)

    weights = weights_accountant.get_global_weights()
    model.set_weights(weights)

    return history, model


def evaluate_federated_cnn(comm_round, test_data=None, test_labels=None, df=None, model=None, weights_accountant=None,
                           history=None, people=None, optimizer=None, loss=None, metrics=None):
    """
    Evaluate the global CNN.

    :param df:
    :param metrics:
    :param loss:
    :param optimizer:
    :param test_data:                       numpy array
    :param test_labels:                     numpy array
    :param comm_round:                      int, specifying the current communication round
    :param model:                           Tensorflow graph
    :param weights_accountant:              WeightsAccountant object
    :param history:                         History object, used for logging
    :param people:                          numpy array, of len test_labels, containing client numbers

    :return:
        history                             history object
    """

    if model is None:
        model = init_global_model(optimizer, loss, metrics)
    if weights_accountant is not None:
        weights = weights_accountant.get_global_weights()
    else:
        weights = np.load(FEDERATED_GLOBAL_WEIGHTS, allow_pickle=True)
    model.set_weights(weights)

    history = painCNN.evaluate_pain_cnn(model, comm_round, test_data, test_labels, df, history, people, loss)

    return history

# ---------------------------------------------- End Federated Learning -------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #
