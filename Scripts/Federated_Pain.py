import os
import time

import numpy as np
import tensorflow as tf

from Scripts import Centralized_Pain as cP
from Scripts import Model_Architectures as mA
from Scripts import Data_Loader_Functions as dL
from Scripts import Print_Functions as Output
from Scripts import Reset_Model as Reset
from Scripts.Keras_Custom import EarlyStopping
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


def get_localized_layers(exclude_name, model):
    personal_layers = []
    for layer in model.layers:
        if exclude_name not in layer.name and layer.trainable and len(layer.get_weights()) > 0:
            personal_layers.extend(layer.get_weights())
    return np.array(personal_layers)


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Federated Learning ---------------------------------------------- #


def train_client_model(client, local_epochs, model, train_data, train_labels, test_data, test_labels, test_people,
                       all_labels, weights_accountant, personalization):
    """
    Utility function training a simple CNN for 1 client in a federated setting and adding those weights to the
    weights_accountant. Call this function in a federated loop that then makes the weights_accountant average the
    weights to send to a global model.

    :param all_labels:
    :param test_people:
    :param test_labels:
    :param test_data:
    :param personalization:
    :param client:                      int, index for a specific client to be trained
    :param local_epochs:                      int, local epochs to be trained
    :param model:                       Tensorflow Graph
    :param train_data:                  numpy array, partitioned into a number of clients
    :param train_labels:                numpy array, partitioned into a number of clients
    :param weights_accountant:          WeightsAccountant object
    :return:
    """

    old_weights = model.get_weights()
    model, history = cP.train_cnn('federated', model, local_epochs, train_data, train_labels, test_data, test_labels,
                                  test_people, all_labels)

    # Check if only the convolutional layers should be averaged
    if personalization:
        localized_weights = get_localized_layers('conv2d', model)
        weights_accountant.append_localized_layer(client, localized_weights)
        weights = np.concatenate([layer.get_weights() for layer in model.layers if 'conv2d' in layer.name and
                                  layer.trainable])
    else:
        weights = model.get_weights()

    # Only append weights for updating the model, if there was an update
    update_check = [np.array_equal(w_1, w_2) for w_1, w_2 in zip(old_weights, weights)]
    print("Number of layers: {} | Number of layers to update: {}"
          .format(len(update_check), len(update_check) - sum(update_check)))
    if not all(update_check):
        weights_accountant.append_local_weights(weights)

    return history


def client_learning(model, client, local_epochs, train_data, train_labels, test_data, test_labels, test_people,
                    all_labels, weights_accountant, personalization):
    """
    Initializes a client model and kicks off the training of that client by calling "train_client_model".

    :param all_labels:
    :param test_people:
    :param test_labels:
    :param test_data:
    :param personalization:
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
    if personalization and weights_accountant.is_localized(client):
        weights = weights_accountant.get_client_weights(client)
    model.set_weights(weights)

    # Train local model and store weights to folder
    return train_client_model(client, local_epochs, model, train_data, train_labels, test_data, test_labels,
                              test_people, all_labels, weights_accountant, personalization)


def communication_round(model, clients, train_data, train_labels, test_data, test_labels, test_people, all_labels,
                        local_epochs, weights_accountant, num_participating_clients, personalization):
    """
    One round of communication between a 'server' and the 'clients'. Each client 'downloads' a global model and trains
    a local model, updating its weights locally. When all clients have updated their weights, they are 'uploaded' to
    the server and averaged.

    :param all_labels:
    :param test_people:
    :param test_labels:
    :param test_data:
    :param personalization:
    :param model:                           Tensorflow Graph
    :param clients:                         int, number of clients globally available, or array
    :param train_data:                      numpy array
    :param train_labels:                    numpy array
    :param local_epochs:                    int, number of epochs each client will train in a given communication round
    :param weights_accountant:              WeightsAccountant object
    :param num_participating_clients:       int, number of participating clients in a given communication round
    :return:
    """

    # Select clients to participate in communication round
    client_arr = np.unique(clients)
    if type(num_participating_clients) is int:
        clients = create_client_index_array(client_arr, num_participating_clients)

    train_data, train_labels = dL.split_data_into_clients_dict(clients, train_data, train_labels)

    if test_data is not None:
        test_data, test_labels, test_all_labels, test_people, all_labels = \
            dL.split_data_into_clients_dict(test_people, test_data, test_labels, all_labels, test_people, all_labels)

    # Train each client
    history = {}
    for client in client_arr:
        client_data = train_data.get(client)
        client_labels = train_labels.get(client)
        if test_data is not None:
            client_test_data = test_data.get(client)
            client_test_labels = test_labels.get(client)
            client_test_people = test_people.get(client)
            client_all_labels = all_labels.get(client)
        else:
            client_test_data, client_test_labels, client_test_people, client_all_labels = [None] * 4
        client_id = client[0, 0].astype(int) if type(client) is np.ndarray else client

        Output.print_client_id(client_id)
        results = client_learning(model, client_id, local_epochs, client_data, client_labels, client_test_data,
                                  client_test_labels, client_test_people, client_all_labels, weights_accountant,
                                  personalization)

        for key, val in results.items():
            history.setdefault(key, []).extend(val)

    # Pop general metrics from history as these are duplicated with client level metrics, and thus not meaningful
    for metric in model.metrics_names:
        history.pop(metric, None)
        history.pop("val_" + metric, None)

    # Average all local updates and store them as new 'global weights'
    if weights_accountant is not None:
        weights_accountant.average_local_weights()

    return history


def federated_learning(model, global_epochs, train_data, train_labels, test_data, test_labels, loss, people, clients,
                       local_epochs, participating_clients, optimizer, metrics, model_type, personalization,
                       all_labels):
    """
    Train a federated model for a specified number of rounds until convergence.

    :param all_labels:
    :param personalization:
    :param model_type:
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
    :param local_epochs:                    int, number of epochs each client will train in a given communication round
    :param participating_clients:       int, number of participating clients in a given communication round
    :param people:                          numpy array, of length test_labels, used to enable individual metric logging

    :return:
        history                             pandas data-frame, contains the history of loss & accuracy values off all
                                            communication rounds
    """

    # Create history object and callbacks
    history = {key: [] for key in ['loss', 'accuracy', 'TP', 'TN', 'FN', 'FP']}
    for key in history.keys():
        for client in clients:
            history["subject_" + str(client) + "_" + key]: []

    early_stopping = EarlyStopping(patience=5)

    # Initialize a random global model and store the weights
    if model is None:
        model = init_global_model(optimizer, loss, metrics, model_type=model_type)

    # Initialize weights accountant
    weights = model.get_weights()
    weights_accountant = WeightsAccountant(weights)

    # Start communication rounds and save the results of each round to the data frame
    for comm_round in range(global_epochs):
        Output.print_communication_round(comm_round + 1)
        results = communication_round(model, clients, train_data, train_labels, test_data, test_labels, people,
                                      all_labels,
                                      local_epochs, weights_accountant, participating_clients, personalization)

        # Only get the first of the local epochs
        for key in results.keys():
            first_epoch = results[key].pop(0)
            history.setdefault(key, []).append(first_epoch)

        # Append None, if no entry was present
        for key in history.keys():
            if key not in results and "subject_" in key:
                history[key].append(None)

        # Evaluate the global model
        weights = weights_accountant.get_global_weights()
        model.set_weights(weights)
        train_metrics = model.metrics_names
        train_history = dict(zip(train_metrics, model.evaluate(train_data, train_labels)))
        for key_1, val_1 in train_history.items():
            history.setdefault(key_1, []).append(val_1)

        validation_metrics = ["val_" + metric for metric in model.metrics_names]
        test_history = dict(zip(validation_metrics, model.evaluate(test_data, test_labels)))
        for key_2, val_2 in test_history.items():
            history.setdefault(key_2, []).append(val_2)

        # Early stopping
        if early_stopping(test_history['val_loss']):
            print("Early Stopping, Communication round {}".format(comm_round))
            break

        print("\n\nHISTORY")
        print(history)

    weights = weights_accountant.get_client_weights() if personalization else weights_accountant.get_global_weights()
    model.set_weights(weights)

    return history, model

# def evaluate_federated_cnn(comm_round, test_data=None, test_labels=None, df=None, model=None, weights_accountant=None,
#                            history=None, people=None, optimizer=None, loss=None, metrics=None, model_type='CNN',
#                            predict_gen=None, personalization=False):
#     """
#     Evaluate the global CNN.
#
#     :param personalization:
#     :param predict_gen:
#     :param model_type:
#     :param df:
#     :param metrics:
#     :param loss:
#     :param optimizer:
#     :param test_data:                       numpy array
#     :param test_labels:                     numpy array
#     :param comm_round:                      int, specifying the current communication round
#     :param model:                           Tensorflow graph
#     :param weights_accountant:              WeightsAccountant object
#     :param history:                         History object, used for logging
#     :param people:                          numpy array, of len test_labels, containing client numbers
#
#     :return:
#         history                             history object
#     """
#
#     if model is None:
#         model = init_global_model(optimizer, loss, metrics, model_type)
#
#     if personalization:
#         clients = df['Person'].unique() if df is not None else np.unique(people)
#         conv_weights = weights_accountant.get_global_weights()
#         for client in clients:
#             client_df = df[df['Person'] == client] if df is not None else None
#             if client_df is not None:
#                 predict_gen = cP.set_up_data_generator(client_df, model.name, shuffle=False, balanced=False,
#                                                        gen_type="Client {}".format(client))
#
#             personal_weights = weights_accountant.get_localized_layers(client)
#             weights = np.concatenate((conv_weights, personal_weights))
#             model.set_weights(weights)
#             history = cP.evaluate_pain_cnn(model, comm_round, test_data, test_labels, predict_gen, history, people,
#                                            loss, client_df)
#
#     else:
#         weights = weights_accountant.get_global_weights()
#         model.set_weights(weights)
#         history = cP.evaluate_pain_cnn(model, comm_round, test_data, test_labels, predict_gen, history, people, loss,
#                                        df)
#     return history

# ---------------------------------------------- End Federated Learning -------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #
