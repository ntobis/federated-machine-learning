import argparse
import os
import sys
import traceback

import sklearn

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time

import pandas as pd
import numpy as np
import tensorflow as tf
from twilio.rest import Client

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

from Scripts import Print_Functions as Output
from Scripts import Data_Loader_Functions as dL
from Scripts import Centralized_Pain as cP
from Scripts import Federated_Pain as fP
from Scripts import Model_Architectures as mA

pd.set_option('display.max_columns', 500)

ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS = os.path.join(ROOT, 'Models')
CENTRAL_PAIN_MODELS = os.path.join(ROOT, "Models", "Pain", "Centralized")
RESULTS = os.path.join(ROOT, 'Results')


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Utility Functions ----------------------------------------------- #


class GoogleCloudMonitor:
    def __init__(self, project='smooth-drive-248209', zone='us-west1-b', instance='federated-imperial-vm'):
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

    def send_message(self, msg=None):
        body = ['Sir, this is Google speaking. Your Federated model trained like a boss. Google out.',
                "Nico you garstige Schlange. What a training session. I'm going to sleep",
                "Wow, what a ride. Training complete.",
                "This was wild. But I trained like crazy. We're done here."]
        if msg is None:
            msg = np.random.choice(body)
        self.messages.create(to=self.args.receiver, from_=self.args.sender, body=msg)


def training_setup(seed):
    # Training setup
    print("GPU Available: ", tf.test.is_gpu_available())
    # tf.debugging.set_log_device_placement(True)
    tf.random.set_seed(seed)
    np.random.seed(seed)


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


def runner_centralized_pain(dataset, experiment, model, train_data=None, train_labels=None, test_data=None,
                            test_labels=None, df=None, epochs=5, people=None, loss=None, session=None, evaluate=True):
    """
    Sets up a centralized CNN that trains on a specified dataset. Saves the results to CSV.

    :param session:
    :param df:
    :param evaluate:
    :param loss:
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

    model, history = cP.train_cnn(model, epochs=epochs, train_data=train_data,
                                  train_labels=train_labels, test_data=test_data,
                                  test_labels=test_labels, df=df, people=people, evaluate=evaluate,
                                  loss=loss, session=session)

    # Save full model
    folder = os.path.join(CENTRAL_PAIN_MODELS, time.strftime("%Y-%m-%d"))
    if not os.path.isdir(folder):
        os.mkdir(folder)
    f_name = time.strftime("%Y-%m-%d-%H%M%S") + r"_{}_{}.h5".format(dataset, experiment)
    model.save(os.path.join(folder, f_name))

    # Save Final Results
    folder = os.path.join(RESULTS, time.strftime("%Y-%m-%d") + "_{}_{}".format(dataset,
                                                                               experiment.split("_shard")[0]))
    if not os.path.isdir(folder):
        os.mkdir(folder)
    suffix = r'_final_results_individual.csv' if people is not None else r'_final_results_aggregate.csv'
    f_name = time.strftime("%Y-%m-%d-%H%M%S") + "_{}_{}".format(dataset, experiment) + suffix
    history.to_csv(os.path.join(folder, f_name))

    return model


def runner_federated_pain(dataset, experiment, clients=None, train_data=None, train_labels=None, test_data=None,
                          test_labels=None, df=None, rounds=5, epochs=1, participants=None, people=None, model=None,
                          optimizer=None, loss=None, metrics=None, session=False, evaluate=True, model_type='CNN'):
    """
    Sets up a federated CNN that trains on a specified dataset. Saves the results to CSV.

    :param model_type:
    :param evaluate:
    :param df:
    :param session:
    :param metrics:
    :param loss:
    :param optimizer:
    :param clients:                 int, the maximum number of clients participating in a communication round
    :param dataset:                 string, name of the dataset to be used, e.g. "MNIST"
    :param experiment:              string, the type of experimental setting to be used, e.g. "CLIENTS"
    :param train_data:              numpy array, the train data
    :param train_labels:            numpy array, the train labels
    :param test_data:               numpy array, the test data
    :param test_labels:             numpy array, the test labels
    :param rounds:                  int, number of communication rounds that the federated clients average results for
    :param epochs:                  int, number of epochs that the client CNN trains for
    :param participants:            participants in a given communications round
    :param people:                  numpy array of len test_labels, enabling individual client metrics
    :param model:                   A compiled tensorflow model
    :return:
    """

    # Reset federated model
    fP.reset_federated_model()

    # Train Model
    history, model = fP.federated_learning(communication_rounds=rounds,
                                           num_of_clients=clients,
                                           train_data=train_data,
                                           train_labels=train_labels,
                                           test_data=test_data,
                                           test_labels=test_labels,
                                           local_epochs=epochs,
                                           num_participating_clients=participants,
                                           people=people,
                                           model=model,
                                           optimizer=optimizer,
                                           loss=loss,
                                           metrics=metrics,
                                           session=session,
                                           df=df,
                                           evaluate=evaluate,
                                           model_type=model_type
                                           )

    # Save history for plotting
    folder = os.path.join(RESULTS, time.strftime("%Y-%m-%d") + "_{}_{}".format(dataset,
                                                                               experiment.split("_shard")[0]))
    if not os.path.isdir(folder):
        os.mkdir(folder)
    suffix = r'_final_results_individual.csv' if people is not None else r'_final_results_aggregate.csv'
    f_name = time.strftime("%Y-%m-%d-%H%M%S") + "_{}_{}".format(dataset, experiment) + suffix
    history.to_csv(os.path.join(folder, f_name))

    # Save model
    folder = os.path.join(MODELS, "Pain", "Federated", time.strftime("%Y-%m-%d"))
    if not os.path.isdir(folder):
        os.mkdir(folder)
    f_name = time.strftime("%Y-%m-%d-%H%M%S") + r"_{}_{}.h5".format(dataset, experiment)
    model.save(os.path.join(folder, f_name))

    return model


# ---------------------------------------------- End Experiment Runners -------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Experiments - PAIN ---------------------------------------------- #

def experiment_pain_centralized(dataset, experiment, rounds, shards=None, pretraining=True, cumulative=True,
                                optimizer=None, loss=None, metrics=None, model_type='CNN'):
    # Define data paths
    group_1_train_path = os.path.join(ROOT, "Data", "Augmented Data", "Flexible Augmentation", "group_1")
    group_2_train_path = os.path.join(ROOT, "Data", "Augmented Data", "Flexible Augmentation", "group_2_train")
    group_2_test_path = os.path.join(ROOT, "Data", "Augmented Data", "Flexible Augmentation", "group_2_test")

    # Define labels for training
    person = 0  # Labels: [person, session, culture, frame, pain, Trans_1, Trans_2]
    pain = 4

    # Initialize OneHotEncoder
    enc = sklearn.preprocessing.OneHotEncoder(sparse=False, categories='auto')

    # Initialize random model
    model = mA.build_model((215, 215, 1), model_type)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Perform pre-training on group 1
    if pretraining:
        # Load data
        train_data, train_labels = dL.load_pain_data(group_1_train_path)

        # Prepare labels for training and evaluation
        train_labels_ord = train_labels[:, pain].astype(np.int)
        train_labels_binary = dL.reduce_pain_label_categories(train_labels_ord, max_pain=1)
        train_labels_binary = enc.fit_transform(train_labels_binary.reshape(len(train_labels_binary), 1))

        # Train
        model = runner_centralized_pain(dataset, experiment + "_shard-0.00", model=model, train_data=train_data,
                                        train_labels=train_labels_binary, epochs=rounds, loss=loss, evaluate=False)
        # Free memory
        del train_data

    if shards is not None:
        # Load the test data
        test_data, test_labels = dL.load_pain_data(group_2_test_path)
        test_labels_ordinal = test_labels[:, pain].astype(np.int)
        test_labels_binary = dL.reduce_pain_label_categories(test_labels_ordinal, max_pain=1)
        test_labels_people = test_labels[:, person].astype(np.int)
        test_labels_binary = enc.fit_transform(test_labels_binary.reshape(len(test_labels_binary), 1))

        # Load group 2 training data
        group_2_train_data, group_2_train_labels = dL.load_pain_data(group_2_train_path)
        group_2_train_labels_ordinal = group_2_train_labels[:, pain].astype(np.int)
        group_2_train_labels_binary = dL.reduce_pain_label_categories(group_2_train_labels_ordinal, max_pain=1)
        group_2_train_labels_binary = enc.fit_transform(group_2_train_labels_binary.reshape(
            len(group_2_train_labels_binary), 1))
        group_2_train_labels_people = group_2_train_labels[:, person].astype(np.int)

        # Split group 2 training data into shards
        group_2_train_data, group_2_train_labels_binary, group_2_split_people = dL.split_data_into_shards(
            array=[group_2_train_data,
                   group_2_train_labels_binary,
                   group_2_train_labels_people],
            split=shards,
            cumulative=cumulative)

        # Train on group 2 shards and evaluate performance
        for percentage, data, labels, people in zip(shards, group_2_train_data, group_2_train_labels_binary,
                                                    group_2_split_people):
            Output.print_shard(percentage)
            Output.print_shard_summary(labels, people)
            experiment_current = experiment + "_shard-{}".format(percentage)
            model = runner_centralized_pain(dataset, experiment_current, model=model, train_data=data,
                                            train_labels=labels, test_data=test_data, test_labels=test_labels_binary,
                                            df=rounds, people=test_labels_people, loss=loss)

    # Split group 2 training data into sessions
    else:

        # Prepare data generator
        group_2_path = os.path.join(ROOT, "Data", "Augmented Data", "Flexible Augmentation", "group_2")
        df = dL.create_pain_df(group_2_path)

        # Run Sessions
        for session in df['Session'].unique():
            Output.print_session(session)
            experiment_current = experiment + "_shard-{}".format(session)
            model = runner_centralized_pain(dataset, experiment_current, model=model, df=df, epochs=rounds, loss=loss,
                                            session=session)


def experiment_pain_federated(dataset, experiment, rounds, shards=None, clients=None, model_path=None, pretraining=None,
                              cumulative=True, optimizer=None, loss=None, metrics=None, subjects_per_client=None,
                              local_epochs=1, model_type='CNN'):
    # Define data paths
    group_1_train_path = os.path.join(ROOT, "Data", "Augmented Data", "Flexible Augmentation", "group_1")
    group_2_train_path = os.path.join(ROOT, "Data", "Augmented Data", "Flexible Augmentation", "group_2_train")
    group_2_test_path = os.path.join(ROOT, "Data", "Augmented Data", "Flexible Augmentation", "group_2_test")

    # Define labels for training
    person = 0  # Labels: [person, session, culture, frame, pain, Trans_1, Trans_2]
    pain = 4

    # Initialize OneHotEncoder
    enc = sklearn.preprocessing.OneHotEncoder(sparse=False, categories='auto')

    # Perform pre-training on group 1
    if pretraining == 'federated':
        if model_path is not None:
            model = tf.keras.models.load_model(model_path)
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        else:
            # Load data
            train_data, train_labels = dL.load_pain_data(group_1_train_path)

            # Prepare labels for training and evaluation
            train_labels_ordinal = train_labels[:, pain].astype(np.int)
            train_labels_binary = dL.reduce_pain_label_categories(train_labels_ordinal, max_pain=1)
            train_labels_binary = enc.fit_transform(train_labels_binary.reshape(len(train_labels_binary), 1))

            # Split data into clients
            if clients is None:
                client_arr = np.unique(train_labels[:, person])
                train_data, train_labels_binary, all_labels = \
                    dL.split_data_into_clients(len(client_arr), 'person', train_data, train_labels_binary,
                                               train_labels, subjects_per_client=subjects_per_client)

                # If no clients are specified, clients will be separated according to the "person" label
                clients = all_labels
            else:
                train_data, train_labels_binary = dL.split_data_into_clients(clients, 'random', train_data,
                                                                             train_labels_binary)

            # Train
            model = runner_federated_pain(dataset, experiment + "_shard-0.00", clients, train_data,
                                          train_labels_binary, rounds=rounds, evaluate=False, loss=loss,
                                          metrics=metrics, optimizer=optimizer, epochs=local_epochs,
                                          model_type=model_type)

    elif pretraining == 'centralized':
        model = tf.keras.models.load_model(model_path)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    elif pretraining is None:
        model = None
    else:
        raise ValueError("Invalid Argument. Arguments allowed are 'federated', 'centralized', and None. Argument was {}"
                         .format(pretraining))

    if shards is not None:
        # Load test data
        test_data, test_labels = dL.load_pain_data(group_2_test_path)
        test_labels_ordinal = test_labels[:, pain].astype(np.int)
        test_labels_binary = dL.reduce_pain_label_categories(test_labels_ordinal, max_pain=1)
        test_labels_people = test_labels[:, person].astype(np.int)
        test_labels_binary = enc.fit_transform(test_labels_binary.reshape(len(test_labels_binary), 1))

        # Load group 2 training data
        group_2_train_data, group_2_train_labels = dL.load_pain_data(group_2_train_path)
        group_2_train_labels_ordinal = group_2_train_labels[:, pain].astype(np.int)
        group_2_train_labels_binary = dL.reduce_pain_label_categories(group_2_train_labels_ordinal, max_pain=1)
        group_2_train_labels_binary = enc.fit_transform(group_2_train_labels_binary.reshape(
            len(group_2_train_labels_binary), 1))

        # Split group 2 training data into shards
        group_2_train_data, group_2_train_labels_binary, group_2_train_labels = dL.split_data_into_shards(
            array=[group_2_train_data,
                   group_2_train_labels_binary,
                   group_2_train_labels],
            split=shards,
            cumulative=cumulative)

        # Train on group 2 shards and evaluate performance
        for percentage, data, labels, all_labels in zip(shards, group_2_train_data, group_2_train_labels_binary,
                                                        group_2_train_labels):
            Output.print_shard(percentage)
            Output.print_shard_summary(labels, all_labels[:, person])

            # Split data into clients
            client_arr = np.unique(all_labels[:, person])
            data, labels, all_labels = dL.split_data_into_clients(len(client_arr), 'person', data, labels,
                                                                  all_labels=all_labels,
                                                                  subjects_per_client=subjects_per_client)

            experiment_current = experiment + "_shard-{}".format(percentage)
            model = runner_federated_pain(dataset, experiment_current, all_labels, data, labels, test_data,
                                          test_labels_binary, rounds=rounds, people=test_labels_people, model=model,
                                          optimizer=optimizer, loss=loss, metrics=metrics, epochs=local_epochs,
                                          model_type=model_type)

    # Split group 2 into sessions
    else:

        # Prepare df for data generator
        group_2_path = os.path.join(ROOT, "Data", "Augmented Data", "Flexible Augmentation", "group_2")
        df = dL.create_pain_df(group_2_path)

        # Run Sessions
        for session in df['Session'].unique():
            Output.print_session(session)
            experiment_current = experiment + "_shard-{}".format(session)
            model = runner_federated_pain(dataset, experiment_current, df=df, rounds=rounds, epochs=local_epochs,
                                          model=model, optimizer=optimizer, loss=loss, metrics=metrics, session=session,
                                          evaluate=True, model_type=model_type)


# ------------------------------------------------ End Experiments - 3 --------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


def main(seed=123, unbalanced=False, balanced=False, sessions=False, redistribution=False):
    # Setup
    data_loc = os.path.join(ROOT, "Data", "Augmented Data", "Flexible Augmentation")

    # g_monitor = GoogleCloudMonitor()
    twilio = Twilio()

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = ['accuracy']

    # Define shards
    test_shards = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    try:

        # --------------------------------------- UNBALANCED ---------------------------------------#
        if unbalanced:
            if redistribution:
                training_setup(seed)
                dL.prepare_pain_images(data_loc, distribution='unbalanced')

            # Experiment 1 - Unbalanced: Centralized without pre-training
            training_setup(seed)
            Output.print_experiment("1 - Unbalanced: Centralized without pre-training")
            experiment_pain_centralized('PAIN', '1-unbalanced-Centralized-no-pre-training', 30, shards=test_shards,
                                        pretraining=False, cumulative=True, optimizer=optimizer, loss=loss,
                                        metrics=metrics)
            twilio.send_message("Experiment 1 Complete")

            # Experiment 2 - Unbalanced: Centralized with pre-training
            training_setup(seed)
            Output.print_experiment("2 - Unbalanced: Centralized with pre-training")
            experiment_pain_centralized('PAIN', '2-unbalanced-Centralized-pre-training', 30, shards=test_shards,
                                        pretraining=True, cumulative=True, optimizer=optimizer,
                                        loss=loss, metrics=metrics)
            twilio.send_message("Experiment 2 Complete")

            # Experiment 3 - Unbalanced: Federated without pre-training
            training_setup(seed)
            Output.print_experiment("3 - Unbalanced: Federated without pre-training")
            experiment_pain_federated('PAIN', '3-unbalanced-Federated-no-pre-training', 30, shards=test_shards,
                                      clients=None, pretraining=None, cumulative=True, optimizer=optimizer, loss=loss,
                                      metrics=metrics, subjects_per_client=1)
            twilio.send_message("Experiment 3 Complete")

            # Experiment 4 - Unbalanced: Federated with centralized pretraining
            training_setup(seed)
            Output.print_experiment("4 - Unbalanced: Federated with centralized pretraining")
            centralized_model_path = find_newest_model_path(os.path.join(CENTRAL_PAIN_MODELS, "2019-07-31"),
                                                            "shard-0.00.h5")
            experiment_pain_federated('PAIN', '4-unbalanced-Federated-central-pre-training', 30, shards=test_shards,
                                      clients=None, model_path=centralized_model_path, pretraining='centralized',
                                      cumulative=True, optimizer=optimizer, loss=loss, metrics=metrics,
                                      subjects_per_client=1)
            twilio.send_message("Experiment 4 Complete")

            # Experiment 5 - Unbalanced: Federated with federated pretraining
            training_setup(seed)
            Output.print_experiment("5 - Unbalanced: Federated with federated pretraining")
            experiment_pain_federated('PAIN', '5-unbalanced-Federated-federated-pre-training', 30, shards=test_shards,
                                      clients=None, pretraining='federated', cumulative=True,
                                      optimizer=optimizer, loss=loss, metrics=metrics, subjects_per_client=1)
            twilio.send_message("Experiment 5 Complete")

        # --------------------------------------- BALANCED ---------------------------------------#

        if balanced:
            if redistribution:
                training_setup(seed)
                dL.prepare_pain_images(data_loc, distribution='balanced')

            # Experiment 6 - Balanced: Centralized without pre-training
            training_setup(seed)
            Output.print_experiment("6 - Balanced: Centralized without pre-training")
            experiment_pain_centralized('PAIN', '1-balanced-Centralized-no-pre-training', 30, shards=test_shards,
                                        pretraining=False, cumulative=True, optimizer=optimizer, loss=loss,
                                        metrics=metrics)
            twilio.send_message("Experiment 6 Complete")

            # Experiment 7 - Balanced: Centralized with pre-training
            training_setup(seed)
            Output.print_experiment("7 - Balanced: Centralized with pre-training")
            experiment_pain_centralized('PAIN', '2-balanced-Centralized-pre-training', 30, shards=test_shards,
                                        pretraining=True, cumulative=True, optimizer=optimizer,
                                        loss=loss, metrics=metrics)
            twilio.send_message("Experiment 7 Complete")

            # Experiment 8 - Balanced: Federated without pre-training
            training_setup(seed)
            Output.print_experiment("8 - Balanced: Federated without pre-training")
            experiment_pain_federated('PAIN', '3-balanced-Federated-no-pre-training', 30, shards=test_shards,
                                      clients=None, pretraining=None, cumulative=True, optimizer=optimizer, loss=loss,
                                      metrics=metrics, subjects_per_client=1)
            twilio.send_message("Experiment 8 Complete")

            # Experiment 9 - Balanced: Federated with centralized pretraining
            training_setup(seed)
            Output.print_experiment("9 - Balanced: Federated with centralized pretraining")
            centralized_model_path = find_newest_model_path(os.path.join(CENTRAL_PAIN_MODELS, "2019-07-31"),
                                                            "shard-0.00.h5")
            experiment_pain_federated('PAIN', '4-balanced-Federated-central-pre-training', 30, shards=test_shards,
                                      clients=None, model_path=centralized_model_path, pretraining='centralized',
                                      cumulative=True, optimizer=optimizer, loss=loss, metrics=metrics,
                                      subjects_per_client=1)
            twilio.send_message("Experiment 9 Complete")

            # Experiment 10 - Balanced: Federated with federated pretraining
            training_setup(seed)
            Output.print_experiment("10 - Balanced: Federated with federated pretraining")
            experiment_pain_federated('PAIN', '5-balanced-Federated-federated-pre-training', 30,
                                      shards=test_shards, clients=None,
                                      pretraining='federated', cumulative=True,
                                      optimizer=optimizer, loss=loss, metrics=metrics, subjects_per_client=1)
            twilio.send_message("Experiment 10 Complete")

        # --------------------------------------- SESSIONS ---------------------------------------#

        if sessions:
            if redistribution:
                training_setup(seed)
                dL.prepare_pain_images(data_loc, distribution='unbalanced')

            # Experiment 11 - Sessions: Centralized without pre-training
            training_setup(seed)
            Output.print_experiment("11 - Sessions: Centralized without pre-training")
            experiment_pain_centralized('PAIN', '1-sessions-Centralized-no-pre-training', 30, shards=None,
                                        pretraining=False, cumulative=True, optimizer=optimizer, loss=loss,
                                        metrics=metrics, model_type='ResNet')
            twilio.send_message("Experiment 11 Complete")

            # Experiment 12 - Sessions: Centralized with pre-training
            training_setup(seed)
            Output.print_experiment("12 - Sessions: Centralized with pre-training")
            experiment_pain_centralized('PAIN', '2-sessions-Centralized-pre-training', 30, shards=None,
                                        pretraining=True, cumulative=True, optimizer=optimizer,
                                        loss=loss, metrics=metrics)
            twilio.send_message("Experiment 12 Complete")

            # Experiment 13 - Sessions: Federated without pre-training
            training_setup(seed)
            Output.print_experiment("13 - Sessions: Federated without pre-training")
            experiment_pain_federated('PAIN', '3-sessions-Federated-no-pre-training', 30, shards=None,
                                      clients=None, pretraining=None, cumulative=True, optimizer=optimizer, loss=loss,
                                      metrics=metrics, subjects_per_client=1)
            twilio.send_message("Experiment 13 Complete")

            # Experiment 14 - Sessions: Federated with centralized pretraining
            training_setup(seed)
            Output.print_experiment("14 - Sessions: Federated with centralized pretraining")
            centralized_model_path = find_newest_model_path(os.path.join(CENTRAL_PAIN_MODELS, "2019-08-06"),
                                                            "shard-0.00.h5")
            experiment_pain_federated('PAIN', '4-sessions-Federated-central-pre-training', 30, shards=None,
                                      clients=None, model_path=centralized_model_path, pretraining='centralized',
                                      cumulative=True, optimizer=optimizer, loss=loss, metrics=metrics,
                                      subjects_per_client=1)

            twilio.send_message("Experiment 14 Complete")

            # Experiment 15 - Sessions: Federated with federated pretraining
            training_setup(seed)
            Output.print_experiment("15 - Sessions: Federated with federated pretraining")
            experiment_pain_federated('PAIN', '5-sessions-Federated-federated-pre-training', 30, shards=None,
                                      clients=None, pretraining='federated', cumulative=True,
                                      optimizer=optimizer, loss=loss, metrics=metrics, subjects_per_client=1)
            twilio.send_message("Experiment 15 Complete")

        twilio.send_message()

    except Exception as e:
        twilio.send_message("Attention, an error occurred:\n{}".format(e)[:1000])
        traceback.print_tb(e.__traceback__)
        print(e)

    # Notify that training is complete and shut down Google server
    g_monitor.shutdown()


if __name__ == '__main__':
    main(seed=123, unbalanced=False, balanced=False, sessions=True, redistribution=False)
