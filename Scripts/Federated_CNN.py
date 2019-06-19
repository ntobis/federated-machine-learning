import os

import numpy as np
import tensorflow as tf

import Scripts.Centralized_CNN as cNN
import Scripts.Print_Functions as Print

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
    train_data = np.array_split(train_data, num_of_clients)
    train_labels = np.array_split(train_labels, num_of_clients)

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


def init_global_model():
    """
    Initializes a global "server-side" model.

    :return:
        model                       tensorflow-graph
    """

    with open(FEDERATED_GLOBAL_MODEL) as json_file:
        json_config = json_file.read()
    model = models.model_from_json(json_config)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    weights = model.get_weights()

    if not os.path.isfile(FEDERATED_GLOBAL_WEIGHTS):
        np.save(FEDERATED_GLOBAL_WEIGHTS, weights)

    return model


def average_local_weights():
    """
    Heart of the federated learning algorithm.
    1. Loads all weights from each client and each layer into an array
    2. Convert the array into a numpy array and average across the array
    3. Store the array as the new global set of weights

    :return:
        average_weights             n-Dimensional numpy array, where 'n' is the number of layers in the network
    """

    layer_stack = []
    for file in os.listdir(FEDERATED_LOCAL_WEIGHTS_PATH):
        path = os.path.join(FEDERATED_LOCAL_WEIGHTS_PATH, file)
        layers = np.load(path, allow_pickle=True)
        layer_stack.append(layers)
    layer_stack = np.array(layer_stack).T

    average_weights = np.mean(layer_stack, axis=1)
    np.save(FEDERATED_GLOBAL_WEIGHTS, average_weights)
    return average_weights


# ---------------------------------------------- End Utility Functions --------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


def communication_round(num_of_clients, train_data, train_labels, num_participating_clients=None):
    """
    One round of communication between a 'server' and the 'clients'. Each client 'downloads' a global model and trains
    a local model, updating its weights locally. When all clients have updated their weights, they are 'uploaded' to
    the server and averaged.

    :param num_of_clients:                  int, number of clients globally available
    :param train_data:                      numpy array
    :param train_labels:                    numpy array
    :param num_participating_clients:       int, number of participating clients in a given communication round
    :return:
    """

    # Select clients to participate in communication round
    clients = create_client_index_array(num_of_clients, num_participating_clients)

    for client in clients:
        Print.print_client_id(client)

        # Initialize model structure and load weights
        model = init_global_model()
        weights = np.load(FEDERATED_GLOBAL_WEIGHTS, allow_pickle=True)
        model.set_weights(weights)

        # Train local model and store weights to folder
        model = cNN.train_cnn(model, train_data[client], train_labels[client], epochs=5)
        weights = model.get_weights()
        np.save(FEDERATED_LOCAL_WEIGHTS.format(client), weights)

    # Average all local updates and store them as new 'global weights'
    average_local_weights()


def federated_learning(communication_rounds, num_of_clients, train_data, train_labels, test_data, test_labels):
    """
    Train a federated model for a specified number of rounds until convergence.

    :param communication_rounds:            int, number of times the global weights will be updated
    :param num_of_clients:                  int, number of clients globally available
    :param train_data:                      numpy array
    :param train_labels:                    numpy array
    :param test_data:                       numpy array
    :param test_labels:                     numpy array
    :return:
    """

    for _ in range(communication_rounds):
        Print.print_communication_round(_ + 1)
        communication_round(num_of_clients, train_data, train_labels)
        evaluate_federated_cnn(test_data, test_labels)


def evaluate_federated_cnn(test_data, test_labels):
    """
    Evaluate the global CNN.

    :param test_data:                       numpy array
    :param test_labels:                     numpy array
    :return:
        test_loss                           float
        test_acc                            float
    """

    model = init_global_model()
    weights = np.load(FEDERATED_GLOBAL_WEIGHTS, allow_pickle=True)
    model.set_weights(weights)
    test_loss, test_acc = cNN.evaluate_cnn(model, test_data, test_labels)
    Print.print_loss_accuracy(test_acc, test_loss)
    return test_loss, test_acc


def main(clients, plotting=False, evaluating=True, max_samples=None):
    """
    Main function including a number of flags that can be set

    :param clients:             int (specifying number of participating clients)
    :param plotting:            bool
    :param evaluating:          bool
    :param max_samples:         int

    :return:

    """

    # Load data
    train_images, train_labels, test_images, test_labels = cNN.load_mnist_data()

    if max_samples:
        train_images = train_images[:max_samples]
        train_labels = train_labels[:max_samples]

    # Split training data
    train_data, train_labels = split_data_into_clients(clients, train_images, train_labels)

    # Display data
    if plotting:
        cNN.display_images(train_images, train_labels)

    # Build initial model
    model = cNN.build_cnn(input_shape=(28, 28, 1))

    # Save initial model
    json_config = model.to_json()
    with open(FEDERATED_GLOBAL_MODEL, 'w') as json_file:
        json_file.write(json_config)

    # Train Model
    federated_learning(10, clients, train_data, train_labels, test_images, test_labels)

    # Evaluate model
    if evaluating:
        evaluate_federated_cnn(test_images, test_labels)

    # # Plot Accuracy and Loss
    # if plotting:
    #     plot_accuracy(model)
    #     plot_loss(model)


if __name__ == '__main__':
    main(clients=10)
