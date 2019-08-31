import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import backend as K

from Scripts import Data_Loader_Functions as dL
from Scripts import Keras_Custom as kC
from Scripts import Print_Functions as Output
from Scripts.Keras_Custom import EarlyStopping

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------ Paths ----------------------------------------------------- #

ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS = os.path.join(ROOT, 'Models')
FEDERATED_LOCAL_WEIGHTS_PATH = os.path.join(MODELS, 'Pain', 'Federated', 'Federated Weights')
LR_FACTOR = 10

# ---------------------------------------------------- End Paths --------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


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


def change_layer_status(model, criterion, operation):
    layers_modified = 0
    print("{} the following layers:".format(operation.capitalize()))
    for layer in model.layers:
        if criterion in layer.name:
            if operation.lower() == 'freeze':
                layer.trainable = False
            elif operation.lower() == 'unfreeze':
                layer.trainable = True
            else:
                raise ValueError("'Operation' accepts only parameters 'freeze' and 'unfreeze'. "
                                 "{} was given".format(operation))
            print(layer.name)
            layers_modified += 1
    return layers_modified


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Federated Learning ---------------------------------------------- #


def train_client_model(client, local_epochs, model, train_data, train_labels, val_data, val_labels, val_people,
                       val_all_labels, weights_accountant, individual_validation):
    """
    Utility function training a simple CNN for 1 client in a federated setting and adding those weights to the
    weights_accountant. Call this function in a federated loop that then makes the weights_accountant average the
    weights to send to a global model.

    :param individual_validation:
    :param val_all_labels:
    :param val_people:
    :param val_labels:
    :param val_data:
    :param client:                      int, index for a specific client to be trained
    :param local_epochs:                      int, local epochs to be trained
    :param model:                       Tensorflow Graph
    :param train_data:                  numpy array, partitioned into a number of clients
    :param train_labels:                numpy array, partitioned into a number of clients
    :param weights_accountant:          WeightsAccountant object
    :return:
    """

    weights = model.get_weights()
    model, history = train_cnn('federated', model, local_epochs, train_data, train_labels, val_data, val_labels,
                               val_people, val_all_labels, individual_validation)

    # If there was an update to the layers, add the update to the weights accountant
    unchanged_layers = [np.array_equal(w_1, w_2) for w_1, w_2 in zip(weights, model.get_weights())]
    print("Layers: {} | Layers to update: {}".format(len(unchanged_layers),
                                                     len(unchanged_layers) - sum(unchanged_layers)))
    if not all(unchanged_layers):
        weights_accountant.update_client_weights(model, client)

    return history


def client_learning(model, client, local_epochs, train_data, train_labels, val_data, val_labels, val_people,
                    val_all_labels, weights_accountant, individual_validation):
    """
    Initializes a client model and kicks off the training of that client by calling "train_client_model".

    :param individual_validation:
    :param val_all_labels:
    :param val_people:
    :param val_labels:
    :param val_data:
    :param model:                       Tensorflow graph
    :param client:                      int, index for a specific client to be trained, or array tb converted to int
    :param local_epochs:                int, local epochs to be trained
    :param train_data:                  numpy array, partitioned into a number of clients
    :param train_labels:                numpy array, partitioned into a number of clients
    :param weights_accountant:          WeightsAccountant object
    :return:
    """

    # Get all client specific data points
    client_data = train_data.get(client)
    client_labels = train_labels.get(client)
    if val_data is not None:
        client_val_data = val_data.get(client)
        client_val_labels = val_labels.get(client)
        client_val_people = val_people.get(client)
        client_val_all_labels = val_all_labels.get(client)
    else:
        client_val_data, client_val_labels, client_val_people, client_val_all_labels = [None] * 4

    # Initialize model structure and load weights
    weights_accountant.apply_client_weights(model, client)

    # Train local model and store weights to folder
    return train_client_model(client, local_epochs, model, client_data, client_labels, client_val_data,
                              client_val_labels, client_val_people, client_val_all_labels, weights_accountant,
                              individual_validation)


def communication_round(model, clients, train_data, train_labels, train_people, val_data, val_labels, val_people,
                        val_all_labels, local_epochs, weights_accountant, individual_validation, local_operation):
    """
    One round of communication between a 'server' and the 'clients'. Each client 'downloads' a global model and trains
    a local model, updating its weights locally. When all clients have updated their weights, they are 'uploaded' to
    the server and averaged.

    :param train_people:
    :param local_operation:
    :param individual_validation:
    :param val_all_labels:
    :param val_people:
    :param val_labels:
    :param val_data:
    :param model:                           Tensorflow Graph
    :param clients:                         int, number of clients globally available, or array
    :param train_data:                      numpy array
    :param train_labels:                    numpy array
    :param local_epochs:                    int, number of epochs each client will train in a given communication round
    :param weights_accountant:              WeightsAccountant object
    :return:
    """

    # Split train and validation data into clients
    train_data, train_labels = dL.split_data_into_clients_dict(train_people, train_data, train_labels)
    if val_data is not None:
        val_data, val_labels, val_people, val_all_labels = \
            dL.split_data_into_clients_dict(val_people, val_data, val_labels, val_people, val_all_labels)

    # Train each client
    history = {}
    for client in clients:
        Output.print_client_id(client)
        results = client_learning(model, client, local_epochs, train_data, train_labels, val_data, val_labels,
                                  val_people, val_all_labels, weights_accountant, individual_validation)

        # Append each client's results to the history dictionary
        for key, val in results.items():
            history.setdefault(key, []).extend(val)

    # Pop general metrics from history as these are duplicated with client metrics, e.g. 'loss' == 'subject_43_loss'
    for metric in model.metrics_names:
        history.pop(metric, None)
        history.pop("val_" + metric, None)

    # If there is localization (e.g. the last layer of the model is not being averaged, indicated by less "shared
    # weights" compared to total "default weights"), then we adapt local models to the new shared layers
    if local_operation == 'localized_learning':

        # Average all updates marked as "global"
        weights_accountant.federated_averaging(layer_type='global')

        # Decrease the learning rate for local adaptation only
        K.set_value(model.optimizer.lr, K.get_value(model.optimizer.lr) / LR_FACTOR)

        # Freeze the global layers
        change_layer_status(model, 'global', 'freeze')

        # Reconnect the Convolutional layers
        for client in clients:
            Output.print_client_id(client)
            client_learning(model, client, local_epochs, train_data, train_labels, val_data, val_labels,
                            val_people, val_all_labels, weights_accountant, individual_validation)

        # Unfreeze the global layers
        change_layer_status(model, 'global', 'unfreeze')

        # Increase the learning rate again
        K.set_value(model.optimizer.lr, K.get_value(model.optimizer.lr) * LR_FACTOR)

    elif local_operation == 'local_models':
        print("No federated averaging.")
        pass

    elif local_operation == 'global_averaging':
        weights_accountant.federated_averaging()

    else:
        raise ValueError('local_operation only accepts "global_averaging", "localized_learning", and "local_models"'
                         ' as arguments. "{}" was given.'.format(local_operation))

    return history


def federated_learning(model, global_epochs, train_data, train_labels, train_people, val_data, val_labels, val_people,
                       val_all_labels, clients, local_epochs, individual_validation, local_operation,
                       weights_accountant):
    """
    Train a federated model for a specified number of rounds until convergence.

    :param train_people:
    :param weights_accountant:
    :param local_operation:
    :param individual_validation:
    :param val_all_labels:
    :param model:
    :param global_epochs:            int, number of times the global weights will be updated
    :param clients:                  int, number of clients globally available
    :param train_data:                      numpy array
    :param train_labels:                    numpy array
    :param val_data:                       numpy array
    :param val_labels:                     numpy array
    :param local_epochs:                    int, number of epochs each client will train in a given communication round
    :param val_people:              numpy array, of length val_labels, used to enable individual metric logging

    :return:
        history                             pandas data-frame, contains the history of loss & accuracy values off all
                                            communication rounds
    """

    # Create history object and callbacks
    history = {}
    keys = [metric for metric in model.metrics_names]
    for key in keys:
        history[key] = []
        for client in clients:
            history["subject_" + str(client) + "_" + key] = []

    early_stopping = EarlyStopping(patience=5)

    # Start communication rounds and save the results of each round to the data frame
    for comm_round in range(global_epochs):
        Output.print_communication_round(comm_round + 1)
        results = communication_round(model, clients, train_data, train_labels, train_people, val_data, val_labels,
                                      val_people, val_all_labels, local_epochs, weights_accountant,
                                      individual_validation, local_operation)

        # Only get the first of the local epochs
        for key in results.keys():
            first_epoch = results[key].pop(0)
            history.setdefault(key, []).append(first_epoch)

        # Append None, if no entry was present
        for key in history.keys():
            if key not in results and "subject_" in key:
                history[key].append(None)

        # Evaluate the global model
        train_metrics = model.metrics_names
        val_metrics = ["val_" + metric for metric in model.metrics_names]

        if local_operation == 'localized_learning' or local_operation == 'local_models':
            split_train_data, split_train_labels = dL.split_data_into_clients_dict(train_people, train_data,
                                                                                   train_labels)
            split_val_data, split_val_labels = dL.split_data_into_clients_dict(val_people, val_data, val_labels)

            train_history = {}
            val_history = {}
            for client in clients:
                client_train_data, client_train_labels = split_train_data.get(client), split_train_labels.get(client)
                weights_accountant.apply_client_weights(model, client)

                train_results = dict(zip(train_metrics, model.evaluate(client_train_data, client_train_labels)))
                for key, value in train_results.items():
                    train_history.setdefault(key, []).append(value)

            for val_person in np.unique(val_people):
                client_val_data, client_val_labels = split_val_data.get(val_person), \
                                                       split_val_labels.get(val_person)
                weights_accountant.apply_client_weights(model, val_person)
                val_results = dict(zip(val_metrics, model.evaluate(client_val_data, client_val_labels)))
                for key, value in val_results.items():
                    val_history.setdefault(key, []).append(value)

            temp_history = calculate_weighted_average(train_history, model.metrics_names)
            temp_history.update(calculate_weighted_average(val_history, model.metrics_names, 'val_'))
            for key, value in temp_history.items():
                history.setdefault(key, []).extend(value)

        elif local_operation == 'global_averaging':
            weights_accountant.apply_default_weights(model)

            train_results = dict(zip(train_metrics, model.evaluate(train_data, train_labels)))
            for key, value in train_results.items():
                history.setdefault(key, []).append(value)

            val_results = dict(zip(val_metrics, model.evaluate(val_data, val_labels)))
            for key, value in val_results.items():
                history.setdefault(key, []).append(value)

        else:
            raise ValueError('local_operation only accepts "global_averaging", "localized_learning", and "local_models"'
                             ' as arguments. "{}" was given.'.format(local_operation))

        # Early stopping
        if early_stopping(weights_accountant.get_client_weights(), history.get('val_loss')[-1]):
            print("Early Stopping, Communication round {}".format(comm_round))
            weights = early_stopping.return_best_weights()
            weights_accountant.set_client_weights(weights)
            break

        weights_accountant.print_client_update()

    return history, model


def calculate_weighted_average(history, metrics, prefix=''):
    weight_columns = ['false_positives', 'false_negatives', 'true_positives', 'true_negatives']
    weight_columns = [prefix + col for col in weight_columns]
    avg_columns = [prefix + col for col in metrics if prefix + col not in weight_columns]
    df = pd.DataFrame(history)
    df['X'] = df[weight_columns].sum(axis=1)
    df_avg = pd.DataFrame(df[avg_columns].mul(df['X'], axis=0).divide(df['X'].sum()).sum()).T
    df_avg = pd.concat((df_avg, pd.DataFrame(df[weight_columns].sum()).T), axis=1)
    return df_avg.to_dict('list')


def train_cnn(algorithm, model, epochs, train_data, train_labels, val_data, val_labels, val_people, val_all_labels,
              individual_validation):

    # Create callbacks
    history_cb = None
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto',
                                                      baseline=None, restore_best_weights=True)
    callbacks = [early_stopping]
    # Create validation sets
    validation_data = (val_data, val_labels) if val_data is not None and algorithm is not 'federated' else None

    if individual_validation:
        history_cb = add_additional_validation_callback(callbacks, val_data, val_labels, val_people, val_all_labels)

    # Train and evaluate
    validation_split = 0.2 if val_data is None and algorithm == 'centralized' else None

    history = model.fit(train_data, train_labels, epochs=epochs, batch_size=32, use_multiprocessing=True,
                        validation_split=validation_split, validation_data=validation_data, callbacks=callbacks)

    return model, history_cb.history if history_cb is not None else history.history


def add_additional_validation_callback(callbacks, val_data, val_labels, val_people, val_all_labels):
    _, val_data_split, val_labels_split, val_people_split = dL.split_data_into_labels(0, val_all_labels, False,
                                                                                         val_data, val_labels,
                                                                                         val_people)
    validation_sets = [(val_data, val_labels, 'subject_{}'.format(person[0]))
                       for val_data, val_labels, person in
                       zip(val_data_split, val_labels_split, val_people_split)]
    history_cb = kC.AdditionalValidationSets(validation_sets)
    callbacks.insert(0, history_cb)
    return history_cb
