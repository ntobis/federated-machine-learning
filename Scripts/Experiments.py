import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time

import pandas as pd
import numpy as np
import tensorflow as tf
from twilio.rest import Client

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

from Scripts import Centralized_CNN as cNN
from Scripts import Federated_CNN as fed_CNN
from Scripts import Print_Functions as Output
from Scripts import Data_Loader_Functions as dL
from Scripts import Centralized_Pain_CNN as painCNN
from Scripts import Federated_Transfer_Learning_CNN as fedTransCNN

pd.set_option('display.max_columns', 500)


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Utility Functions ----------------------------------------------- #
class GoogleCloudMonitor:
    def __init__(self, project='federated-learning-244811', zone='us-west1-b', instance='federated-learning'):
        # Google Credentials Set Up
        self.credentials = GoogleCredentials.get_application_default()
        self.service = discovery.build('compute', 'v1', credentials=self.credentials)

        # Project ID for this request.
        self.project = project

        # The name of the zone for this request.
        self.zone = zone

        # Name of the instance resource to stop.
        self.instance = instance

    def shutdown(self):
        request = self.service.instances().stop(project=self.project, zone=self.zone, instance=self.instance)
        return request.execute()


class Twilio(Client):
    def __init__(self):
        # Parse Commandline Arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--sms_acc", help="Enter Twilio Account Here")
        parser.add_argument("--sms_pw", help="Enter Twilio Password Here")
        parser.add_argument("--sender", help="Sender Number")
        parser.add_argument("--receiver", help="Sender Number")
        self.args = parser.parse_args()
        super(Twilio, self).__init__(self.args.sms_acc, self.args.sms_pw)

    def send_training_complete_message(self, msg=None):
        body = ['Sir, this is Google speaking. Your Federated model trained like a boss. Google out.',
                "Nico you garstige Schlange. What a training session. I'm going to sleep",
                "Wow, what a ride. Training complete.",
                "This was wild. But I trained like crazy. We're done here."]
        if msg is None:
            msg = np.random.choice(body)
        self.messages.create(to=self.args.receiver, from_=self.args.sender, body=msg)


def training_setup():
    # Training setup
    print("GPU Available: ", tf.test.is_gpu_available())
    # tf.debugging.set_log_device_placement(True)
    tf.random.set_seed(123)
    np.random.seed(123)


def move_results(experiment, date, keys):
    """
    Utility function used to move results into a specified folder with a name of the format:
    EXPERIMENT DATE KEYS

    :param experiment:                  string, name of the experiment
    :param date:                        string, date of the experiment
    :param keys:                        list, type of experiment, e.g. number of clients: [2, 5, 10, 100]
    :return:
    """

    experiment_path = os.path.join(cNN.RESULTS, experiment + " " + date + " " + str(keys))
    if not os.path.isdir(experiment_path):
        os.mkdir(experiment_path)
    for elem in os.listdir(cNN.RESULTS):
        if os.path.isfile(os.path.join(cNN.RESULTS, elem)):
            old_loc = os.path.join(cNN.RESULTS, elem)
            new_loc = os.path.join(cNN.RESULTS, experiment_path, elem)
            os.rename(old_loc, new_loc)


def combine_results(experiment, keys, sub_folder=None):
    """
    Combines the results from various experiments into one data frame that can be used for joint plotting of metrics.

    :param experiment:                  string, the type of experimental setting to be used, e.g. "CLIENTS"
    :param keys:                        array, the different experiments, e.g. number of clients [2, 5, 10]
    :param sub_folder:                  string, where in the "Results" folder the results should be stored

    :return:
        history:                        DataFrame, holds the results of all experiments, passed to a plotting function
    """

    # Open most recent history files
    if sub_folder:
        files = os.listdir(os.path.join(cNN.RESULTS, sub_folder))
        files = [os.path.join(os.path.join(cNN.RESULTS, sub_folder), file) for file in files]
    else:
        files = os.listdir(cNN.RESULTS)
        files = [os.path.join(cNN.RESULTS, file) for file in files]

    # Combine Results
    sorted_files = sorted(files, key=os.path.getctime)
    centralized = pd.read_csv(sorted_files[0], index_col=0)
    history = centralized
    for idx in range(len(keys)):
        federated = pd.read_csv(sorted_files[-idx - 1], index_col=0)
        federated = federated.rename(index=str,
                                     columns={"Federated Train Loss": "Federated Train Loss {} {}".format(
                                         experiment,
                                         keys[idx]),
                                         "Federated Train Accuracy": "Federated Train Accuracy {} {}".format(
                                             experiment,
                                             keys[idx]),
                                         "Federated Test Loss": "Federated Test Loss {} {}".format(
                                             experiment,
                                             keys[idx]),
                                         "Federated Test Accuracy": "Federated Test Accuracy {} {}".format(
                                             experiment,
                                             keys[idx])})
        history = pd.concat([history.reset_index(drop=True), federated.reset_index(drop=True)], axis=1)

    return history


def find_newest_model_path(path, sub_string):
    files = []
    for dir_path, dirname, filenames in os.walk(path):
        files.extend([os.path.join(dir_path, f_name) for f_name in filenames])
    pre_train = [file for file in files if sub_string in file]
    pre_train.sort(key=os.path.getmtime)
    return pre_train[-1]


# ---------------------------------------------- End Utility Functions --------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Experiment Runners ---------------------------------------------- #


def plot_results(dataset, experiment, keys, date, suffix, move=False):
    """
    Sets the parameters for the plotting function, and calls the plotting function to plott loss and accuracy over
    multiple epochs/communication rounds.

    :param dataset:                 string, name of the dataset to be used, e.g. "MNIST"
    :param experiment:              string, the type of experimental setting to be used, e.g. "CLIENTS"
    :param keys:                    array, the different experiments, e.g. number of clients [2, 5, 10]
    :param date:                    string, date to be used for folder naming
    :param suffix:                  string, additional information to be added to the folder name
    :param move:                    bool, set to true if results are still in general "Results" folder
    :return:
    """

    if move:
        move_results(experiment, time.strftime("%Y-%m-%d"), keys)

    sub_folder = experiment + " " + date + " " + suffix
    history = combine_results(experiment, keys, sub_folder)

    # Plot Accuracy
    params = Output.PlotParams(
        dataset=dataset,
        experiment=experiment,
        metric='Accuracy',
        title='Model Accuracy {} {}'.format(experiment, suffix),
        x_label='Federated Comm. Round/Centralized Epoch',
        y_label='Accuracy',
        legend_loc='lower right',
        num_format="{:5.1%}",
        max_epochs=None,
        label_spaces=4,
        suffix=suffix
    )
    Output.plot_joint_metric(history, params)

    # Plot Loss
    params = Output.PlotParams(
        dataset=dataset,
        experiment=experiment,
        metric='Loss',
        title='Model Loss {} {}'.format(experiment, suffix),
        x_label='Federated Comm. Round/Centralized Epoch',
        y_label='Loss',
        legend_loc='upper right',
        num_format="{:5.2f}",
        max_epochs=None,
        label_spaces=4,
        suffix=suffix
    )
    Output.plot_joint_metric(history, params)


def runner_centralized_mnist(dataset, experiment, train_data, train_labels, test_data, test_labels, epochs=5):
    """
    Sets up a centralized CNN that trains on a specified dataset. Saves the results to CSV.

    :param dataset:                 string, name of the dataset to be used, e.g. "MNIST"
    :param experiment:              string, the type of experimental setting to be used, e.g. "CLIENTS"
    :param train_data:              numpy array, the train data
    :param train_labels:            numpy array, the train labels
    :param test_data:               numpy array, the test data
    :param test_labels:             numpy array, the test labels
    :param epochs:                  int, number of epochs that the centralized CNN trains for
    :return:
    """

    # Train Centralized CNN
    centralized_model = cNN.build_cnn(input_shape=train_data[0].shape)
    centralized_model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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


def runner_centralized_pain(dataset, experiment, train_data, train_labels, test_data, test_labels, epochs=5,
                            model=None, people=None):
    """
    Sets up a centralized CNN that trains on a specified dataset. Saves the results to CSV.

    :param model:
    :param people:
    :param dataset:                 string, name of the dataset to be used, e.g. "MNIST"
    :param experiment:              string, the type of experimental setting to be used, e.g. "CLIENTS"
    :param train_data:              numpy array, the train data
    :param train_labels:            numpy array, the train labels
    :param test_data:               numpy array, the test data
    :param test_labels:             numpy array, the test labels
    :param epochs:                  int, number of epochs that the centralized CNN trains for
    :return:
    """

    # Train Centralized CNN
    if model is None:
        centralized_model = painCNN.build_cnn(input_shape=train_data[0].shape)
        centralized_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        centralized_model = model

    centralized_model, history = painCNN.train_cnn(centralized_model, epochs=epochs, train_data=train_data,
                                                   train_labels=train_labels, test_data=test_data,
                                                   test_labels=test_labels, people=people, evaluate=False)

    # Save full model
    folder = os.path.join(painCNN.CENTRAL_PAIN_MODELS, time.strftime("%Y-%m-%d"))
    if not os.path.isdir(folder):
        os.mkdir(folder)
    f_name = time.strftime("%Y-%m-%d-%H%M%S") + r"_{}_{}.h5".format(dataset, experiment)
    centralized_model.save(os.path.join(folder, f_name))

    # Save Final Results
    folder = os.path.join(cNN.RESULTS, time.strftime("%Y-%m-%d") + "_{}_{}".format(dataset,
                                                                                   experiment.split("_shard")[0]))
    if not os.path.isdir(folder):
        os.mkdir(folder)
    suffix = r'_final_results_individual.csv' if people is not None else r'_final_results_aggregate.csv'
    f_name = time.strftime("%Y-%m-%d-%H%M%S") + "_{}_{}".format(dataset, experiment) + suffix
    history.to_csv(os.path.join(folder, f_name))

    return centralized_model


def runner_federated_mnist(clients, dataset, experiment, train_data, train_labels, test_data, test_labels,
                           rounds=5, epochs=1, split='random', participants=None):
    """
    Sets up a federated CNN that trains on a specified dataset. Saves the results to CSV.

    :param clients:                 int, the maximum number of clients participating in a communication round
    :param dataset:                 string, name of the dataset to be used, e.g. "MNIST"
    :param experiment:              string, the type of experimental setting to be used, e.g. "CLIENTS"
    :param train_data:              numpy array, the train data
    :param train_labels:            numpy array, the train labels
    :param test_data:               numpy array, the test data
    :param test_labels:             numpy array, the test labels
    :param rounds:                  int, number of communication rounds that the federated clients average results for
    :param epochs:                  int, number of epochs that the client CNN trains for
    :param split:                   Determine if split should occur randomly
    :param participants:            participants in a given communications round
    :return:
    """

    train_data, train_labels = dL.split_data_into_clients(clients, split, train_data, train_labels)

    # Reset federated model
    fed_CNN.reset_federated_model()

    # Train Model
    history = fed_CNN.federated_learning(communication_rounds=rounds,
                                         num_of_clients=clients,
                                         train_data=train_data,
                                         train_labels=train_labels,
                                         test_data=test_data,
                                         test_labels=test_labels,
                                         epochs=epochs,
                                         num_participating_clients=participants
                                         )

    # Save history for plotting
    file = time.strftime("%Y-%m-%d-%H%M%S") + r"_Federated_{}_{}_rounds_{}_clients_{}.csv".format(dataset, experiment,
                                                                                                  rounds, clients)
    history = history.rename(index=str, columns={"Train Loss": "Federated Train Loss",
                                                 "Train Accuracy": "Federated Train Accuracy",
                                                 "Test Loss": "Federated Test Loss",
                                                 "Test Accuracy": "Federated Test Accuracy"})
    history.to_csv(os.path.join(cNN.RESULTS, file))


def runner_federated_pain(clients, dataset, experiment, train_data, train_labels, test_data, test_labels,
                          rounds=5, epochs=1, split='random', participants=None, people=None, model=None):
    """
    Sets up a federated CNN that trains on a specified dataset. Saves the results to CSV.

    :param clients:                 int, the maximum number of clients participating in a communication round
    :param dataset:                 string, name of the dataset to be used, e.g. "MNIST"
    :param experiment:              string, the type of experimental setting to be used, e.g. "CLIENTS"
    :param train_data:              numpy array, the train data
    :param train_labels:            numpy array, the train labels
    :param test_data:               numpy array, the test data
    :param test_labels:             numpy array, the test labels
    :param rounds:                  int, number of communication rounds that the federated clients average results for
    :param epochs:                  int, number of epochs that the client CNN trains for
    :param split:                   Determine if split should occur randomly
    :param participants:            participants in a given communications round
    :param people:                  numpy array of len test_labels, enabling individual client metrics
    :param model:                   A compiled tensorflow model
    :return:
    """

    train_data, train_labels = dL.split_data_into_clients(clients, split, train_data, train_labels)

    # Reset federated model
    fedTransCNN.reset_federated_model()

    # Train Model
    history, model = fedTransCNN.federated_learning(communication_rounds=rounds,
                                                    num_of_clients=clients,
                                                    train_data=train_data,
                                                    train_labels=train_labels,
                                                    test_data=test_data,
                                                    test_labels=test_labels,
                                                    epochs=epochs,
                                                    num_participating_clients=participants,
                                                    people=people,
                                                    model=model
                                                    )

    # Save history for plotting
    folder = os.path.join(cNN.RESULTS, time.strftime("%Y-%m-%d") + "_{}_{}".format(dataset,
                                                                                   experiment.split("_shard")[0]))
    if not os.path.isdir(folder):
        os.mkdir(folder)
    suffix = r'_final_results_individual.csv' if people is not None else r'_final_results_aggregate.csv'
    f_name = time.strftime("%Y-%m-%d-%H%M%S") + "_{}_{}_clients-{}".format(dataset, experiment, clients) + suffix
    history.to_csv(os.path.join(folder, f_name))

    # Save model
    folder = os.path.join(cNN.MODELS, "Pain", "Federated", time.strftime("%Y-%m-%d"))
    if not os.path.isdir(folder):
        os.mkdir(folder)
    f_name = time.strftime("%Y-%m-%d-%H%M%S") + r"_{}_{}.h5".format(dataset, experiment)
    model.save(os.path.join(folder, f_name))

    return model


# ---------------------------------------------- End Experiment Runners -------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Experiments - MNIST --------------------------------------------- #


def experiment_1_number_of_clients(dataset, experiment, rounds, clients):
    """
    First experiment conducted. Experimenting with varying the number of clients used in a federated setting.

    :param dataset:                 string, name of the dataset to be used, e.g. "MNIST"
    :param experiment:              string, the type of experimental setting to be used, e.g. "CLIENTS"
    :param rounds:                  int, number of communication rounds that the federated clients average results for
    :param clients:                 int_array, the maximum number of clients participating in a communication round
    :return:
    """

    # Load data
    train_data, train_labels, test_data, test_labels, dataset = dL.load_data(dataset)

    # Perform Experiments
    for client_num in clients:
        runner_federated_mnist(client_num, dataset, experiment, train_data, train_labels, test_data, test_labels,
                               rounds)

    runner_centralized_mnist(dataset, experiment, train_data, train_labels, test_data, test_labels, rounds)


def experiment_2_limited_digits(dataset, experiment, rounds, digit_array):
    """
    Second experiment conducted. Experimenting with varying the number of MNIST digits used in a centralized and
    federated setting. The number of clients is held constant at 10.

    :param dataset:                 string, name of the dataset to be used, e.g. "MNIST"
    :param experiment:              string, the type of experimental setting to be used, e.g. "CLIENTS"
    :param rounds:                  int, number of communication rounds that the federated clients average results for
    :param digit_array:             array of arrays, specifying the digits to be used in a given experiment,
                                    e.g. [[0,5],[0,2,5]]
    :return:
    """

    # Load data
    train_data, train_labels, test_data, test_labels, dataset = dL.load_data(dataset)

    # Perform Experiments
    for digit in digit_array:
        experiment = experiment + "_" + str(digit)
        train_data_filtered = train_data[np.in1d(train_labels, digit)]
        train_labels_filtered = train_labels[np.in1d(train_labels, digit)]
        test_data_filtered = test_data[np.in1d(test_labels, digit)]
        test_labels_filtered = test_labels[np.in1d(test_labels, digit)]

        runner_federated_mnist(10, dataset, experiment, train_data_filtered, train_labels_filtered, test_data_filtered,
                               test_labels_filtered, rounds)

        runner_centralized_mnist(dataset, experiment, train_data_filtered, train_labels_filtered, test_data_filtered,
                                 test_labels_filtered, rounds)


def experiment_3_add_noise(dataset, experiment, rounds, std_devs):
    """
    Third experiment conducted. Experimenting with varying the noise added to the MNIST data in a centralized and a
    federated setting. The number of clients is held constant at 10.

    :param dataset:                 string, name of the dataset to be used, e.g. "MNIST"
    :param experiment:              string, the type of experimental setting to be used, e.g. "CLIENTS"
    :param rounds:                  int, number of communication rounds that the federated clients average results for
    :param std_devs:                float_array, the standard deviation of gaussian noise to be added to the dataset
    :return:
    """

    # Load data
    train_data, train_labels, test_data, test_labels, dataset = dL.load_data(dataset)

    # Perform Experiments
    for std_dv in std_devs:
        this_experiment = experiment + "_" + str(std_dv)
        train_data_noise = train_data + np.random.normal(loc=0, scale=std_dv, size=train_data.shape)
        Output.display_images(train_data_noise, train_labels)
        runner_federated_mnist(10, dataset, this_experiment, train_data_noise, train_labels, test_data, test_labels,
                               rounds)

        runner_centralized_mnist(dataset, this_experiment, train_data_noise, train_labels, test_data, test_labels,
                                 rounds)


def experiment_4_split_digits(dataset, experiment, rounds, clients):
    """
    First experiment conducted. Experimenting with varying the number of clients used in a federated setting.

    :param dataset:                 string, name of the dataset to be used, e.g. "MNIST"
    :param experiment:              string, the type of experimental setting to be used, e.g. "CLIENTS"
    :param rounds:                  int, number of communication rounds that the federated clients average results for
    :param clients:                 int_array, the maximum number of clients participating in a communication round
    :return:
    """

    # Load data
    train_data, train_labels, test_data, test_labels, dataset = dL.load_data(dataset)

    # Perform Experiments
    for client_num in clients:
        runner_federated_mnist(client_num, dataset, experiment, train_data, train_labels, test_data, test_labels,
                               rounds,
                               split='no_overlap')

    runner_centralized_mnist(dataset, experiment, train_data, train_labels, test_data, test_labels, rounds)


def experiment_5_split_digits_with_overlap(dataset, experiment, rounds, clients):
    """
    First experiment conducted. Experimenting with varying the number of clients used in a federated setting.

    :param dataset:                 string, name of the dataset to be used, e.g. "MNIST"
    :param experiment:              string, the type of experimental setting to be used, e.g. "CLIENTS"
    :param rounds:                  int, number of communication rounds that the federated clients average results for
    :param clients:                 int_array, the maximum number of clients participating in a communication round
    :return:
    """

    # Load data
    train_data, train_labels, test_data, test_labels, dataset = dL.load_data(dataset)

    # Perform Experiments
    for client_num in clients:
        runner_federated_mnist(client_num, dataset, experiment, train_data, train_labels, test_data, test_labels,
                               rounds,
                               split='overlap', participants=10)

    runner_centralized_mnist(dataset, experiment, train_data, train_labels, test_data, test_labels, rounds)


# ---------------------------------------------- End Experiments - MNIST ------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Experiments - PAIN ---------------------------------------------- #

def experiment_pain_centralized(dataset, experiment, rounds, shards=None, pretraining=True, cumulative=True):
    # Define data paths
    group_1_train_path = os.path.join(cNN.ROOT, "Data", "Augmented Data", "Flexible Augmentation", "group_1")
    group_2_train_path = os.path.join(cNN.ROOT, "Data", "Augmented Data", "Flexible Augmentation", "group_2_train")
    group_2_test_path = os.path.join(cNN.ROOT, "Data", "Augmented Data", "Flexible Augmentation", "group_2_test")

    # Define labels for training
    person = 0 # Labels: [person, session, culture, frame, pain, Trans_1, Trans_2]
    session = 1
    pain = 4

    # Load test data
    test_data, test_labels = dL.load_pain_data(group_2_test_path)
    test_labels_ordinal = test_labels[:, pain].astype(np.int)
    test_labels_binary = dL.reduce_pain_label_categories(test_labels_ordinal, max_pain=1)
    test_labels_people = test_labels[:, person].astype(np.int)

    # Perform pre-training on group 1
    if pretraining:
        # Load data
        train_data, train_labels = dL.load_pain_data(group_1_train_path)

        # Prepare labels for training and evaluation
        train_labels_ord = train_labels[:, pain].astype(np.int)
        train_labels_bin = dL.reduce_pain_label_categories(train_labels_ord, max_pain=1)

        # Train
        model = runner_centralized_pain(dataset, experiment + "_shard-0.00", train_data, train_labels_bin, test_data,
                                        test_labels_binary, rounds, people=test_labels_people)
    else:
        # Initialize random model
        model = painCNN.build_cnn(test_data[0].shape)
        model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Load group 2 training data
    group_2_train_data, group_2_train_labels = dL.load_pain_data(group_2_train_path)
    group_2_train_labels_ordinal = group_2_train_labels[:, pain].astype(np.int)
    group_2_train_labels_binary = dL.reduce_pain_label_categories(group_2_train_labels_ordinal, max_pain=1)

    # Split group 2 training data into shards
    if shards is not None:
        group_2_train_data, group_2_train_labels_binary = dL.split_data_into_shards(group_2_train_data,
                                                                                    group_2_train_labels_binary,
                                                                                    shards,
                                                                                    cumulative)
        # Train on group 2 shards and evaluate performance
        for percentage, data, labels in zip(shards, group_2_train_data, group_2_train_labels_binary):
            if percentage > 0.05:
                break

            Output.print_shard(percentage)
            experiment_current = experiment + "_shard-{}".format(percentage)
            model = runner_centralized_pain(dataset, experiment_current, data, labels, test_data, test_labels_binary,
                                            rounds, model=model, people=test_labels_people)

    else:
        group_2_train_data, group_2_train_labels = dL.split_data_into_labels(session, group_2_train_data,
                                                                             group_2_train_labels, cumulative)




def experiment_pain_federated(dataset, experiment, rounds, shards, clients, model_path=None, pretraining=None,
                              cumulative=True):
    # Define data paths
    group_1_train_path = os.path.join(cNN.ROOT, "Data", "Augmented Data", "Flexible Augmentation", "group_1")
    group_2_train_path = os.path.join(cNN.ROOT, "Data", "Augmented Data", "Flexible Augmentation", "group_2_train")
    group_2_test_path = os.path.join(cNN.ROOT, "Data", "Augmented Data", "Flexible Augmentation", "group_2_test")

    # Define labels for training
    label = 4  # Labels: [person, session, culture, frame, pain, Trans_1, Trans_2]
    person = 0

    # Load test data
    test_data, test_labels = dL.load_pain_data(group_2_test_path)
    test_labels_ordinal = test_labels[:, label].astype(np.int)
    test_labels_binary = dL.reduce_pain_label_categories(test_labels_ordinal, max_pain=1)
    test_labels_people = test_labels[:, person].astype(np.int)

    # Perform pre-training on group 1
    if pretraining == 'federated':
        if model_path is not None:
            model = tf.keras.models.load_model(model_path)
            model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        else:
            # Load data
            train_data, train_labels = dL.load_pain_data(group_1_train_path)

            # Prepare labels for training and evaluation
            train_labels_ordinal = train_labels[:, label].astype(np.int)
            train_labels_binary = dL.reduce_pain_label_categories(train_labels_ordinal, max_pain=1)

            # Train
            runner_federated_pain(clients, dataset, experiment + "_shard-0.00", train_data, train_labels_binary,
                                  test_data, test_labels_binary, rounds, people=test_labels_people)

        # Load trained model into memory
        model_path = find_newest_model_path(os.path.join(cNN.MODELS, "Pain", "Federated"), "_shard-0.00.h5")
        model = tf.keras.models.load_model(model_path)
        model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    elif pretraining == 'centralized':
        model = tf.keras.models.load_model(model_path)
        model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    elif pretraining is None:
        model = None
    else:
        raise ValueError("Invalid Argument. Arguments allowed are 'federated', 'centralized', and None. Argument was {}"
                         .format(pretraining))

    # Load group 2 training data
    group_2_train_data, group_2_train_labels = dL.load_pain_data(group_2_train_path)
    group_2_train_labels_ordinal = group_2_train_labels[:, label].astype(np.int)
    group_2_train_labels_binary = dL.reduce_pain_label_categories(group_2_train_labels_ordinal, max_pain=1)

    # Split group 2 training data into shards
    group_2_train_data, group_2_train_labels_binary = dL.split_data_into_shards(group_2_train_data,
                                                                                group_2_train_labels_binary,
                                                                                shards,
                                                                                cumulative)

    # Train on group 2 shards and evaluate performance
    for percentage, data, labels in zip(shards, group_2_train_data, group_2_train_labels_binary):
        Output.print_shard(percentage)
        experiment_current = experiment + "_shard-{}".format(percentage)
        model = runner_federated_pain(clients, dataset, experiment_current, data, labels, test_data, test_labels_binary,
                                      rounds, people=test_labels_people, model=model)


# ------------------------------------------------ End Experiments - 3 --------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


if __name__ == '__main__':
    # Setup functions
    training_setup()
    g_monitor = GoogleCloudMonitor()
    twilio = Twilio()

    # Define shards
    test_shards = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    # Experiment 6 - Centralized without pre-training
    Output.print_experiment("6 - Centralized without pre-training")
    experiment_pain_centralized('PAIN', 'Unbalanced-Centralized-no-pre-training', 30, test_shards, pretraining=False,
                                cumulative=True)
    twilio.send_training_complete_message("Experiment 6 Complete")

    # # Experiment 7 - Centralized with pre-training
    # Output.print_experiment("7 - Centralized with pre-training")
    # experiment_pain_centralized('PAIN', 'Unbalanced-Centralized-pre-training', 30, test_shards, pretraining=True,
    #                             cumulative=True)
    # twilio.send_training_complete_message("Experiment 7 Complete")
    #
    # # Experiment 8 - Federated without pre-training
    # Output.print_experiment("8 - Federated without pre-training")
    # experiment_pain_federated('PAIN', 'Unbalanced-Federated-no-pre-training', 30, test_shards, 12, pretraining=None,
    #                           cumulative=True)
    # twilio.send_training_complete_message("Experiment 8 Complete")
    #
    # # Experiment 9 - Federated with centralized pretraining
    # Output.print_experiment("9 - Federated with centralized pretraining")
    # centralized_model_path = find_newest_model_path(os.path.join(painCNN.CENTRAL_PAIN_MODELS, "2019-07-27"),
    #                                                 "training.h5")
    # experiment_pain_federated('PAIN', 'Unbalanced-Federated-central-pre-training', 30, test_shards, 12,
    #                           model_path=centralized_model_path, pretraining='centralized', cumulative=True)
    # twilio.send_training_complete_message("Experiment 9 Complete")
    #
    # # Experiment 10 - Federated with federated pretraining
    # Output.print_experiment("10 - Federated with federated pretraining")
    # new_model_path = find_newest_model_path(os.path.join(cNN.MODELS, "Pain", "Federated"), "_shard-0.00.h5")
    # experiment_pain_federated('PAIN', 'Unbalanced-Federated-federated-pre-training', 30, test_shards, 12, pretraining='federated',
    #                           cumulative=True, model_path=new_model_path)
    # twilio.send_training_complete_message("Experiment 10 Complete")

    # Notify that training is complete and shut down Google server
    twilio.send_training_complete_message()
    g_monitor.shutdown()
