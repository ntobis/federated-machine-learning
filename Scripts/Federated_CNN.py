import json
import os
import time

import numpy as np
import tensorflow as tf

import Scripts.Centralized_CNN as CNN

models = tf.keras.models  # like 'from tensorflow.keras import models' (PyCharm import issue workaround)

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------ Paths ----------------------------------------------------- #

FEDERATED_GLOBAL_MODEL = os.path.join(CNN.MODELS, "federated_global_model.json")
FEDERATED_GLOBAL_WEIGHTS = os.path.join(CNN.MODELS, "federated_global_weights.npy")
FEDERATED_LOCAL_WEIGHTS_PATH = os.path.join(CNN.MODELS, "Federated Weights")
FEDERATED_LOCAL_WEIGHTS = os.path.join(FEDERATED_LOCAL_WEIGHTS_PATH, "federated_local_weights_client_{}")

# ---------------------------------------------------- End Paths --------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Utility Functions ----------------------------------------------- #

def print_communication_round(com_round):
    print()
    print("----------------------------------------------------------------------------------------------------------------------------------------")
    print("-------------------------------------------------------- Communication Round {} --------------------------------------------------------".format(com_round))


def print_client_id(id):
    print()
    print("----------------------------------------------------------------------------------------------------------------------------------------")
    print("--------------------------------------------------------------- Client {} --------------------------------------------------------------".format(id))


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
    if num_participating_clients:
        clients = np.random.choice(num_of_clients, num_participating_clients)
    else:
        clients = np.arange(num_of_clients)

    return clients


def init_global_model():
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
    clients = create_client_index_array(num_of_clients, num_participating_clients)

    for client in clients:
        print_client_id(client)
        model = init_global_model()
        weights = np.load(FEDERATED_GLOBAL_WEIGHTS, allow_pickle=True)
        model.set_weights(weights)
        model = CNN.train_cnn(model, train_data[client], train_labels[client], epochs=5)
        weights = model.get_weights()
        np.save(FEDERATED_LOCAL_WEIGHTS.format(client), weights)
    average_local_weights()


def federated_learning(communication_rounds, num_of_clients, train_data, train_labels):
    for _ in range(communication_rounds):
        print_communication_round(_+1)
        communication_round(num_of_clients, train_data, train_labels)


def evaluate_federated_cnn(test_images, test_labels):
    model = init_global_model()
    weights = np.load(FEDERATED_GLOBAL_WEIGHTS, allow_pickle=True)
    model.set_weights(weights)
    test_loss, test_acc = CNN.evaluate_cnn(model, test_images, test_labels)
    print("Test Loss: {:5.2f}".format(test_loss))
    print("Test Accuracy: {:5.2f}%".format(100 * test_acc))


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
    train_images, train_labels, test_images, test_labels = CNN.load_MNIST_data()

    if max_samples:
        train_images = train_images[:max_samples]
        train_labels = train_labels[:max_samples]

    # Split training data
    train_data, train_labels = split_data_into_clients(clients, train_images, train_labels)

    # Display data
    if plotting:
        CNN.display_images(train_images, train_labels)

    # Build initial model
    model = CNN.build_cnn(input_shape=(28, 28, 1))

    # Save initial model
    json_config = model.to_json()
    with open(FEDERATED_GLOBAL_MODEL, 'w') as json_file:
        json_file.write(json_config)

    # Train Model
    federated_learning(10, clients, train_data, train_labels)

    # Evaluate model
    if evaluating:
        evaluate_federated_cnn(test_images, test_labels)

    # # Plot Accuracy and Loss
    # if plotting:
    #     plot_accuracy(model)
    #     plot_loss(model)


if __name__ == '__main__':
    main(clients=10)
