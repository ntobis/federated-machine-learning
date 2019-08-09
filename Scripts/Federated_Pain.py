import os

import numpy as np
import tensorflow as tf

from Scripts import Centralized_Pain as cP
from Scripts import Reset_Model as Reset
from Scripts import Print_Functions as Output
from Scripts import Model_Architectures as mA
from Scripts.Weights_Accountant import WeightsAccountant

models = tf.keras.models  # like 'from tensorflow.keras import models' (PyCharm import issue workaround)
# optimizer = tf.keras.optimizers

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------ Paths ----------------------------------------------------- #

ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS = os.path.join(ROOT, 'Models')
FEDERATED_LOCAL_WEIGHTS_PATH = os.path.join(MODELS, 'Pain', 'Federated', 'Federated Weights')

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
    if os.path.isdir(MODELS):
        Reset.remove_files(MODELS)
    else:
        os.mkdir(MODELS)


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


def init_global_model(optimizer, loss, metrics, input_shape=(215, 215, 1), model_type='CNN'):
    """
    Initializes a global "server-side" model.
    :param model_type:
    :param metrics:
    :param loss:
    :param optimizer:
    :param input_shape:             tuple, input shape of one training example (default, MNIST shape)

    :return:
        model                       tensorflow-graph
    """

    # Build the model
    model = mA.build_model(input_shape, model_type)
    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Save initial model

    return model


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Federated Learning ---------------------------------------------- #


def train_client_model(client, local_epochs, model, train_data=None, train_labels=None, df=None, weights_accountant=None,
                       session=None):
    """
    Utility function training a simple CNN for 1 client in a federated setting and adding those weights to the
    weights_accountant. Call this function in a federated loop that then makes the weights_accountant average the
    weights to send to a global model.

    :param df:
    :param session:
    :param client:                      int, index for a specific client to be trained
    :param local_epochs:                      int, local epochs to be trained
    :param model:                       Tensorflow Graph
    :param train_data:                  numpy array, partitioned into a number of clients
    :param train_labels:                numpy array, partitioned into a number of clients
    :param weights_accountant:          WeightsAccountant object
    :return:
    """

    old_weights = model.get_weights()
    if train_data is not None and train_labels is not None:
        model, history = cP.train_cnn(model, local_epochs, train_data[client], train_labels[client], evaluate=False)
    elif df is not None:
        model, history = cP.train_cnn(model, local_epochs, df=df, evaluate=False, session=session)
    else:
        raise ValueError('Need to provide either "train_data" and "train_labels", or "df", None was provided.')
    weights = model.get_weights()

    # Only append weights for updating the model, if there was an update
    if not all([np.array_equal(w_1, w_2) for w_1, w_2 in zip(old_weights, weights)]):
        weights_accountant.append_local_weights(weights)


def client_learning(model, client, local_epochs, train_data=None, train_labels=None, df=None, weights_accountant=None,
                    session=None):
    """
    Initializes a client model and kicks off the training of that client by calling "train_client_model".

    :param session:
    :param df:
    :param model:                       Tensorflow graph
    :param client:                      int, index for a specific client to be trained, or array tb converted to int
    :param local_epochs:                int, local epochs to be trained
    :param train_data:                  numpy array, partitioned into a number of clients
    :param train_labels:                numpy array, partitioned into a number of clients
    :param weights_accountant:          WeightsAccountant object
    :return:
    """

    # Initialize model structure and load weights
    weights = weights_accountant.get_global_weights()
    model.set_weights(weights)

    # Train local model and store weights to folder
    train_client_model(client, local_epochs, model, train_data, train_labels, df, weights_accountant, session=session)


def communication_round(model, clients, train_data=None, train_labels=None, df=None, local_epochs=1,
                        weights_accountant=None,
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
    :param local_epochs:                          int, number of epochs each client will train in a given communication round
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
        df_train = df[df['Person'] == client] if df is not None else None
        client_id = client[0, 0].astype(int) if type(client) is np.ndarray else client

        Output.print_client_id(client_id)
        client_learning(model, idx, local_epochs, train_data, train_labels, df_train, weights_accountant, session)

    # Average all local updates and store them as new 'global weights'
    if weights_accountant is not None:
        weights_accountant.average_local_weights()


def federated_learning(model, global_epochs, train_data, train_labels, test_data, test_labels, df=None, evaluate=True,
                       loss=None, people=None, session=-1, clients=None, local_epochs=1, participating_clients=None,
                       optimizer=None, metrics=None, model_type='CNN'):
    """
    Train a federated model for a specified number of rounds until convergence.

    :param model_type:
    :param evaluate:
    :param df:
    :param session:
    :param metrics:
    :param loss:
    :param optimizer:
    :param model:
    :param global_epochs:            int, number of times the global weights will be updated
    :param clients:                  int, number of clients globally available
    :param train_data:                      numpy array
    :param train_labels:                    numpy array
    :param test_data:                       numpy array
    :param test_labels:                     numpy array
    :param local_epochs:                          int, number of epochs each client will train in a given communication round
    :param participating_clients:       int, number of participating clients in a given communication round
    :param people:                          numpy array, of length test_labels, used to enable individual metric logging

    :return:
        history                             pandas data-frame, contains the history of loss & accuracy values off all
                                            communication rounds
    """

    # Create history object
    history = cP.set_up_history()

    # Initialize a random global model and store the weights
    if model is None:
        model = init_global_model(optimizer, loss, metrics, model_type=model_type)
    weights = model.get_weights()

    # Set up data generators
    df_train, df_test, train_gen, predict_gen = cP.set_up_train_test_generators(df, model, session)

    # Initialize weights accountant
    weights_accountant = WeightsAccountant(weights)

    # Start communication rounds and save the results of each round to the data frame
    for comm_round in range(global_epochs):
        Output.print_communication_round(comm_round + 1)
        communication_round(model, clients, train_data, train_labels, df, local_epochs, weights_accountant,
                            participating_clients, session)
        if evaluate:
            history = evaluate_federated_cnn(comm_round, test_data, test_labels, df_test, model, weights_accountant,
                                             history, people, optimizer, loss, metrics, model_type, predict_gen)

    weights = weights_accountant.get_global_weights()
    model.set_weights(weights)

    return history, model


def evaluate_federated_cnn(comm_round, test_data=None, test_labels=None, df=None, model=None, weights_accountant=None,
                           history=None, people=None, optimizer=None, loss=None, metrics=None, model_type='CNN',
                           predict_gen=None):
    """
    Evaluate the global CNN.

    :param predict_gen:
    :param model_type:
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
        model = init_global_model(optimizer, loss, metrics, model_type)
    weights = weights_accountant.get_global_weights()
    model.set_weights(weights)

    history = cP.evaluate_pain_cnn(model, comm_round, test_data, test_labels, predict_gen, history, people, loss, df)

    return history

# ---------------------------------------------- End Federated Learning -------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #
