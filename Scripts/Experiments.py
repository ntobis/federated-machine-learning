import argparse
import os
import sys
import traceback

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time

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
from Scripts.Keras_Custom import TP, TN, FP, FN

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------ Paths ----------------------------------------------------- #

ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS = os.path.join(ROOT, 'Results')

MODELS = os.path.join(ROOT, 'Models')
CENTRAL_PAIN_MODELS = os.path.join(MODELS, "Pain", "Centralized")
FEDERATED_PAIN_MODELS = os.path.join(MODELS, "Pain", "Federated")

DATA = os.path.join(ROOT, "Data", "Augmented Data", "Flexible Augmentation")
GROUP_1_TRAIN_PATH = os.path.join(DATA, "group_1")
GROUP_2_PATH = os.path.join(DATA, "group_2")
GROUP_2_TRAIN_PATH = os.path.join(DATA, "group_2_train")
GROUP_2_TEST_PATH = os.path.join(DATA, "group_2_test")


# ---------------------------------------------------- End Paths --------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Utility Functions ----------------------------------------------- #


class GoogleCloudMonitor:
    def __init__(self, project='smooth-drive-248209', zone='us-west1-b', instance='federated-imperial-vm'):
        """
        Set up Google Cloud Monitor Instance. This allows to automatically switch off the Google Cloud instance once
        training stops or an error occurs, thus prevents excessive billing.

        :param project:                     string, Google project name
        :param zone:                        string, Google project zone
        :param instance:                    string, Google project instance name
        """

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
        """
        API call to shuts down a Google instance.
        :return:
        """

        request = self.service.instances().stop(project=self.project, zone=self.zone, instance=self.instance)
        return request.execute()


class Twilio(Client):
    def __init__(self):
        """
        Instantiate a Twilio Client that sends text messages when training is complete or an error occurs. Parses login
        credentials from the command line.
        """

        # Parse Commandline Arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--sms_acc", help="Enter Twilio Account Here")
        parser.add_argument("--sms_pw", help="Enter Twilio Password Here")
        parser.add_argument("--sender", help="Sender Number")
        parser.add_argument("--receiver", help="Sender Number")
        self.args = parser.parse_args()
        super(Twilio, self).__init__(self.args.sms_acc, self.args.sms_pw)

    def send_message(self, msg=None):
        """
        Sends a text message.

        :param msg:                 string, message sent. If not specified one of the default messages will be sent.
        :return:
        """

        body = ['Sir, this is Google speaking. Your Federated model trained like a boss. Google out.',
                "Nico you garstige Schlange. What a training session. I'm going to sleep",
                "Wow, what a ride. Training complete.",
                "This was wild. But I trained like crazy. We're done here."]
        if msg is None:
            msg = np.random.choice(body)
        self.messages.create(to=self.args.receiver, from_=self.args.sender, body=msg)


def training_setup(seed):
    """
    Sets seed for experiments and tests if a GPU is available for training.
    :param seed:                    int, seed to be set
    :return:
    """

    # Training setup
    print("GPU Available: ", tf.test.is_gpu_available())
    tf.random.set_seed(seed)
    np.random.seed(seed)


def find_newest_model_path(path, sub_string):
    """
    Utility function, identifying the newest model in a given directory, given that a "sub_string" is found in the model
    file name.

    :param path:                    string, root path from where to start the search
    :param sub_string:              string, substring that must be found in the f_name
    :return:
        file_path:                  string, path to the latest model
    """

    files = []
    for dir_path, dirname, filenames in os.walk(path):
        files.extend([os.path.join(dir_path, f_name) for f_name in filenames])
    pre_train = [file for file in files if sub_string in file]
    pre_train.sort(key=os.path.getmtime)
    return pre_train[-1]


def save_results(dataset, experiment, history, model, folder):
    """
    Utility function saving training results (training history and latest model) to a folder.

    :param dataset:                 string, name of the dataset used for training
    :param experiment:              string, name of the experiment conducted
    :param history:                 Pandas DataFrame, to be saved as CSV
    :param model:                   Tensorflow Graph, to be saved as .h5 file
    :param folder:                  string, absolute path where to save the model
    :return:
    """

    # Save full model
    if not os.path.isdir(folder):
        os.mkdir(folder)
    f_name = time.strftime("%Y-%m-%d-%H%M%S") + r"_{}_{}.h5".format(dataset, experiment)
    model.save(os.path.join(folder, f_name))
    # Save history for plotting
    folder = os.path.join(RESULTS, time.strftime("%Y-%m-%d") + "_{}_{}".format(dataset,
                                                                               experiment.split("_shard")[0]))
    if not os.path.isdir(folder):
        os.mkdir(folder)
    f_name = time.strftime("%Y-%m-%d-%H%M%S") + "_{}_{}.csv".format(dataset, experiment)
    history.to_csv(os.path.join(folder, f_name))


def load_and_prepare_data(path, person, pain, model_type):
    """
    Utility function loading pain image data into memory, and preparing the labels for training.
    Note, this function expects the image files to have the following naming convention:
    "43_0_0_0_2_original_straight.jpg", to be converted into the following label array:
    [person, session, culture, frame, pain_level, transformation_1, transformation_2]

    :param path:                    string, root path to all images to be loaded
    :param person:                  int, index where 'person' appears in the file name converted to an array.
    :param pain:                    int, index where 'pain_level' appears in the file name converted to an array.
    :param model_type:              string, specifying the model_type (CNN, or ResNet)
    :return:
        data:                       4D numpy array, images as numpy array in shape (N, 215, 215, 1)
        labels_binary:              2D numpy array, one-hot encoded labels [no pain, pain] (N, 2)
        train_labels_people:        2D numpy array, only including the "person" label [person] (N, 1)
        labels:                     2D numpy array, all labels as described above (N, 7)
    """

    enc = OneHotEncoder(sparse=False, categories=[range(2)])
    color = 0 if model_type == 'CNN' else 1
    data, labels = dL.load_pain_data(path, color=color)
    labels_ord = labels[:, pain].astype(np.int)
    labels_binary = dL.reduce_pain_label_categories(labels_ord, max_pain=1)
    train_labels_people = labels[:, person].astype(np.int)
    labels_binary = enc.fit_transform(labels_binary.reshape(len(labels_binary), 1))
    return data, labels_binary, train_labels_people, labels


# ---------------------------------------------- End Utility Functions --------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Experiment Runners ---------------------------------------------- #


def run_pretraining(dataset, experiment, local_epochs, loss, metrics, model_path, model_type, optimizer,
                    pretraining, rounds, personalization):
    if model_path is not None:
        print("Loading pre-trained model: {}".format(os.path.basename(model_path)))
        model = tf.keras.models.load_model(model_path, custom_objects={'TP': TP, 'TN': TN, 'FN': FN, 'FP': FP})
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    elif pretraining is 'centralized':
        print("Pre-training a centralized model.")
        model = mA.build_model((215, 215, 1), model_type)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # Prepare labels for training and evaluation
        train_data, train_labels, train_labels_people, raw_labels = load_and_prepare_data(GROUP_1_TRAIN_PATH,
                                                                                          person=0,
                                                                                          pain=4,
                                                                                          model_type=model_type)
        # Train
        model = model_runner(pretraining, dataset, experiment + "_shard-0.00", model=model, rounds=rounds,
                             train_data=train_data, train_labels=train_labels, loss=loss)

    elif pretraining == 'federated':
        print("Pre-training a federated model.")
        # Load data
        train_data, train_labels, train_labels_people, raw_labels = load_and_prepare_data(GROUP_1_TRAIN_PATH, person=0,
                                                                                          pain=4, model_type=model_type)

        # Train
        model = model_runner(pretraining, dataset, experiment + "_shard-0.00", rounds=rounds, train_data=train_data,
                             train_labels=train_labels, loss=loss, clients=train_labels_people,
                             local_epochs=local_epochs, optimizer=optimizer, metrics=metrics, model_type=model_type,
                             personalization=personalization, all_labels=raw_labels)

    elif pretraining is None:
        model = mA.build_model((215, 215, 1), model_type)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    else:
        raise ValueError("Invalid Argument. You must either specify a 'model_path' or provide 'centralized' or "
                         "'federated' as arguments for 'pretraining'.")
    return model


def run_shards(algorithm, cumulative, dataset, experiment, local_epochs, loss, metrics, model, model_type, optimizer,
               rounds, shards, subjects_per_client, personalization):
    # Load test data
    test_data, test_labels, test_labels_people, raw_labels = load_and_prepare_data(GROUP_2_TEST_PATH, person=0,
                                                                                   pain=4, model_type=model_type)
    # Load group 2 training data
    train_data, train_labels, train_labels_people, raw_labels = load_and_prepare_data(GROUP_2_TRAIN_PATH, person=0,
                                                                                      pain=4, model_type=model_type)
    # Split group 2 training data into shards
    train_data, train_labels, raw_labels = dL.split_data_into_shards(
        array=[train_data, train_labels, raw_labels], split=shards, cumulative=cumulative)
    # Train on group 2 shards and evaluate performance
    for percentage, data, labels, all_labels in zip(shards, train_data, train_labels, raw_labels):
        Output.print_shard(percentage)
        Output.print_shard_summary(labels, all_labels[:, 0])
        experiment_current = experiment + "_shard-{}".format(percentage)

        # Split data into clients
        if algorithm is 'federated':
            client_arr = np.unique(all_labels[:, 0])
            data, labels, all_labels = dL.split_data_into_clients('person', data, labels, len(client_arr),
                                                                  all_labels=all_labels,
                                                                  subjects_per_client=subjects_per_client)

        model = model_runner(algorithm, dataset, experiment_current, model=model, rounds=rounds, train_data=data,
                             train_labels=labels, test_data=test_data, test_labels=test_labels,
                             people=test_labels_people, loss=loss, clients=all_labels, local_epochs=local_epochs,
                             optimizer=optimizer, metrics=metrics, model_type=model_type,
                             personalization=personalization)


def run_sessions(algorithm, dataset, experiment, local_epochs, loss, metrics, model, model_type, optimizer, rounds,
                 personalization):
    # Prepare df for data generator
    df = dL.create_pain_df(GROUP_2_PATH)

    # Run Sessions
    train_data, train_labels, train_people, train_all_labels, client_arr = [None] * 5
    for session in df['Session'].unique():
        Output.print_session(session)
        experiment_current = experiment + "_shard-{}".format(session)

        if session > 0:
            df_test = df[df['Session'] == session]
            test_data, test_labels, test_people, test_all_labels = load_and_prepare_data(
                df_test['img_path'].values,
                person=0,
                pain=4,
                model_type=model_type)

            model = model_runner(algorithm, dataset, experiment_current, model=model, rounds=rounds,
                                 train_data=train_data, train_labels=train_labels, test_data=test_data,
                                 test_labels=test_labels, people=test_people,
                                 loss=loss, clients=train_people,
                                 local_epochs=local_epochs,
                                 optimizer=optimizer, metrics=metrics, model_type=model_type,
                                 personalization=personalization, all_labels=test_all_labels)
        df_train = df[df['Session'] <= session]
        df_train = dL.balance_data(df_train, threshold=200)
        train_data, train_labels, train_people, train_all_labels = load_and_prepare_data(
            df_train['img_path'].values,
            person=0, pain=4, model_type=model_type)


def model_runner(algorithm, dataset, experiment, model=None, rounds=5, train_data=None, train_labels=None,
                 test_data=None,
                 test_labels=None, people=None, loss=None, clients=None,
                 local_epochs=1, participants=None, optimizer=None, metrics=None, model_type='CNN',
                 personalization=False, all_labels=None):
    """
    Sets up a federated CNN that trains on a specified dataset. Saves the results to CSV.

    :param all_labels:
    :param personalization:
    :param algorithm:
    :param model_type:
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
    :param local_epochs:                  int, number of epochs that the client CNN trains for
    :param participants:            participants in a given communications round
    :param people:                  numpy array of len test_labels, enabling individual client metrics
    :param model:                   A compiled tensorflow model
    :return:
    """

    if algorithm is 'federated':
        folder = FEDERATED_PAIN_MODELS
        # Reset federated model
        fP.reset_federated_model()

        # Train Model
        history, model = fP.federated_learning(model=model, global_epochs=rounds, train_data=train_data,
                                               train_labels=train_labels, test_data=test_data, test_labels=test_labels,
                                               loss=loss, people=people, clients=clients, local_epochs=local_epochs,
                                               participating_clients=participants, optimizer=optimizer, metrics=metrics,
                                               model_type=model_type, personalization=personalization,
                                               all_labels=all_labels)

    elif algorithm is 'centralized':
        folder = CENTRAL_PAIN_MODELS
        model, history = cP.train_cnn(algorithm=algorithm, model=model, epochs=rounds, train_data=train_data,
                                      train_labels=train_labels, test_data=test_data,
                                      test_labels=test_labels, people=people, all_labels=all_labels)

    else:
        raise ValueError("'runner_type' must be either 'centralized' or 'federated', was: {}".format(algorithm))

    history = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in history.items()]))
    save_results(dataset, experiment, history, model, folder)

    return model


# ---------------------------------------------- End Experiment Runners -------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Experiments - PAIN ---------------------------------------------- #

def experiment_pain(algorithm, dataset, experiment, rounds, shards=None, model_path=None,
                    pretraining=None, cumulative=True, optimizer=None, loss=None, metrics=None,
                    subjects_per_client=None, local_epochs=1, model_type='CNN', personalization=False):
    # Perform pre-training on group 1
    model = run_pretraining(dataset, experiment, local_epochs, loss, metrics, model_path, model_type,
                            optimizer, pretraining, rounds, personalization)

    # If shards are specified, this experiment will be run
    if shards is not None:
        run_shards(algorithm, cumulative, dataset, experiment, local_epochs, loss, metrics, model, model_type,
                   optimizer, rounds, shards, subjects_per_client, personalization)

    # Else, split group 2 into sessions and run this experiment
    else:
        run_sessions(algorithm, dataset, experiment, local_epochs, loss, metrics, model, model_type, optimizer, rounds,
                     personalization)


# ------------------------------------------------ End Experiments - 3 --------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


def main(seed=123, unbalanced=False, balanced=False, sessions=False, redistribution=False):
    # Setup
    data_loc = os.path.join(ROOT, "Data", "Augmented Data", "Flexible Augmentation")

    # g_monitor = GoogleCloudMonitor()
    twilio = Twilio()

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = ['accuracy', TP, TN, FP, FN]

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
            experiment_pain('centralized', 'PAIN', '1-unbalanced-Centralized-no-pre-training', 30, shards=test_shards,
                            pretraining=None, cumulative=True, optimizer=optimizer, loss=loss,
                            metrics=metrics)
            twilio.send_message("Experiment 1 Complete")

            # Experiment 2 - Unbalanced: Centralized with pre-training
            training_setup(seed)
            Output.print_experiment("2 - Unbalanced: Centralized with pre-training")
            experiment_pain('centralized', 'PAIN', '2-unbalanced-Centralized-pre-training', 30, shards=test_shards,
                            pretraining='centralized', cumulative=True, optimizer=optimizer,
                            loss=loss, metrics=metrics)
            twilio.send_message("Experiment 2 Complete")

            # Experiment 3 - Unbalanced: Federated without pre-training
            training_setup(seed)
            Output.print_experiment("3 - Unbalanced: Federated without pre-training")
            experiment_pain("federated", 'PAIN', '3-unbalanced-Federated-no-pre-training', 30, shards=test_shards,
                            pretraining=None, cumulative=True, optimizer=optimizer, loss=loss,
                            metrics=metrics, subjects_per_client=1)
            twilio.send_message("Experiment 3 Complete")

            # Experiment 4 - Unbalanced: Federated with centralized pretraining
            training_setup(seed)
            Output.print_experiment("4 - Unbalanced: Federated with centralized pretraining")
            centralized_model_path = find_newest_model_path(os.path.join(CENTRAL_PAIN_MODELS),
                                                            "shard-0.00.h5")
            experiment_pain("federated", 'PAIN', '4-unbalanced-Federated-central-pre-training', 30, shards=test_shards,
                            model_path=centralized_model_path, pretraining='centralized', cumulative=True,
                            optimizer=optimizer, loss=loss, metrics=metrics, subjects_per_client=1)
            twilio.send_message("Experiment 4 Complete")

            # Experiment 5 - Unbalanced: Federated with federated pretraining
            training_setup(seed)
            Output.print_experiment("5 - Unbalanced: Federated with federated pretraining")
            experiment_pain("federated", 'PAIN', '5-unbalanced-Federated-federated-pre-training', 30,
                            shards=test_shards,
                            pretraining='federated', cumulative=True, optimizer=optimizer, loss=loss,
                            metrics=metrics, subjects_per_client=1)
            twilio.send_message("Experiment 5 Complete")

        # --------------------------------------- BALANCED ---------------------------------------#

        if balanced:
            if redistribution:
                training_setup(seed)
                dL.prepare_pain_images(data_loc, distribution='balanced')

            # Experiment 6 - Balanced: Centralized without pre-training
            training_setup(seed)
            Output.print_experiment("6 - Balanced: Centralized without pre-training")
            experiment_pain('centralized', 'PAIN', '1-balanced-Centralized-no-pre-training', 30, shards=test_shards,
                            pretraining=None, cumulative=True, optimizer=optimizer, loss=loss,
                            metrics=metrics)
            twilio.send_message("Experiment 6 Complete")

            # Experiment 7 - Balanced: Centralized with pre-training
            training_setup(seed)
            Output.print_experiment("7 - Balanced: Centralized with pre-training")
            experiment_pain('centralized', 'PAIN', '2-balanced-Centralized-pre-training', 30, shards=test_shards,
                            pretraining='centralized', cumulative=True, optimizer=optimizer,
                            loss=loss, metrics=metrics)
            twilio.send_message("Experiment 7 Complete")

            # Experiment 8 - Balanced: Federated without pre-training
            training_setup(seed)
            Output.print_experiment("8 - Balanced: Federated without pre-training")
            experiment_pain("federated", 'PAIN', '3-balanced-Federated-no-pre-training', 30, shards=test_shards,
                            pretraining=None, cumulative=True, optimizer=optimizer, loss=loss, metrics=metrics,
                            subjects_per_client=1)
            twilio.send_message("Experiment 8 Complete")

            # Experiment 9 - Balanced: Federated with centralized pretraining
            training_setup(seed)
            Output.print_experiment("9 - Balanced: Federated with centralized pretraining")
            centralized_model_path = find_newest_model_path(os.path.join(CENTRAL_PAIN_MODELS),
                                                            "shard-0.00.h5")
            experiment_pain("federated", 'PAIN', '4-balanced-Federated-central-pre-training', 30, shards=test_shards,
                            model_path=centralized_model_path, pretraining='centralized', cumulative=True,
                            optimizer=optimizer, loss=loss, metrics=metrics, subjects_per_client=1)
            twilio.send_message("Experiment 9 Complete")

            # Experiment 10 - Balanced: Federated with federated pretraining
            training_setup(seed)
            Output.print_experiment("10 - Balanced: Federated with federated pretraining")
            experiment_pain("federated", 'PAIN', '5-balanced-Federated-federated-pre-training', 30, shards=test_shards,
                            pretraining='federated', cumulative=True, optimizer=optimizer, loss=loss, metrics=metrics,
                            subjects_per_client=1)
            twilio.send_message("Experiment 10 Complete")

        # --------------------------------------- SESSIONS ---------------------------------------#

        if sessions:
            if redistribution:
                training_setup(seed)
                dL.prepare_pain_images(data_loc, distribution='sessions')

            # Experiment 11 - Sessions: Centralized without pre-training
            training_setup(seed)
            Output.print_experiment("11 - Sessions: Centralized without pre-training")
            experiment_pain(algorithm='centralized',
                            dataset='PAIN',
                            experiment='1-sessions-Centralized-no-pre-training',
                            rounds=30,
                            shards=None,
                            model_path=None,
                            pretraining=None,
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            model_type='CNN'
                            )
            twilio.send_message("Experiment 11 Complete")

            # Experiment 12 - Sessions: Centralized with pre-training
            training_setup(seed)
            Output.print_experiment("12 - Sessions: Centralized with pre-training")
            experiment_pain(algorithm='centralized',
                            dataset='PAIN',
                            experiment='2-sessions-Centralized-pre-training',
                            rounds=30,
                            shards=None,
                            model_path=None,
                            pretraining='centralized',
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            model_type='CNN'
                            )
            twilio.send_message("Experiment 12 Complete")

            # # Experiment 13 - Sessions: Federated without pre-training
            training_setup(seed)
            Output.print_experiment("13 - Sessions: Federated without pre-training")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='3-sessions-Federated-no-pre-training',
                            rounds=30,
                            shards=None,
                            model_path=None,
                            pretraining=None,
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            subjects_per_client=1,
                            local_epochs=5,
                            model_type='CNN'
                            )
            twilio.send_message("Experiment 13 Complete")
            # Experiment 14 - Sessions: Federated with centralized pretraining
            training_setup(seed)
            Output.print_experiment("14 - Sessions: Federated with centralized pretraining")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='4-sessions-Federated-central-pre-training',
                            rounds=30,
                            shards=None,
                            model_path=find_newest_model_path(CENTRAL_PAIN_MODELS, "shard-0.00.h5"),
                            # model_path=None,
                            pretraining='centralized',
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            subjects_per_client=1,
                            local_epochs=5,
                            model_type='CNN')

            twilio.send_message("Experiment 14 Complete")

            # Experiment 15 - Sessions: Federated with federated pretraining
            training_setup(seed)
            Output.print_experiment("15 - Sessions: Federated with federated pretraining")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='5-sessions-Federated-federated-pre-training',
                            rounds=30,
                            shards=None,
                            pretraining='federated',
                            model_path=None,
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            subjects_per_client=1,
                            local_epochs=5,
                            model_type='CNN'
                            )
            twilio.send_message("Experiment 15 Complete")

            # # Experiment 16 - Sessions: Federated 'BEST' architecture
            # training_setup(seed)
            # Output.print_experiment("16 - Sessions: Best Model Architecture Hypothesis")
            # experiment_pain(algorithm="federated",
            #                 dataset='PAIN',
            #                 experiment='6-sessions-Federated-Best-Model',
            #                 rounds=30,
            #                 shards=None,
            #                 clients=None,
            #                 pretraining='federated',
            #                 model_path=find_newest_model_path(
            #                     os.path.join(FEDERATED_PAIN_MODELS, "Final", "Unbalanced"), "shard-0.00.h5"),
            #                 cumulative=True,
            #                 optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
            #                 loss=loss,
            #                 metrics=metrics,
            #                 subjects_per_client=1,
            #                 local_epochs=1,
            #                 model_type='CNN',
            #                 personalization=False
            #                 )
            # twilio.send_message("Experiment 16 Complete")
            #
            # training_setup(seed)
            # Output.print_experiment("17 - Sessions: Centralized Pre Training")
            # experiment_pain(algorithm='centralized',
            #                 dataset='PAIN',
            #                 experiment='17-sessions-Centralized-pre-training',
            #                 rounds=30,
            #                 shards=None,
            #                 clients=None,
            #                 model_path=find_newest_model_path(CENTRAL_PAIN_MODELS, "shard-0.00.h5"),
            #                 pretraining='centralized',
            #                 cumulative=True,
            #                 optimizer=optimizer,
            #                 loss=loss,
            #                 metrics=metrics,
            #                 model_type='CNN',
            #                 )
            # twilio.send_message("Experiment 17 Complete")

            # training_setup(seed)
            # Output.print_experiment("18 - Sessions: Centralized RMSProp")
            # experiment_pain(algorithm='centralized',
            #                 dataset='PAIN',
            #                 experiment='18-sessions-Centralized-pre-training-RMS',
            #                 rounds=30,
            #                 shards=None,
            #                 clients=None,
            #                 model_path=find_newest_model_path(CENTRAL_PAIN_MODELS, "shard-0.00.h5"),
            #                 pretraining='centralized',
            #                 cumulative=True,
            #                 optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
            #                 loss=loss,
            #                 metrics=metrics,
            #                 model_type='CNN'
            #                 )
            # twilio.send_message("Experiment 18 Complete")
            #
            # training_setup(seed)
            # Output.print_experiment("19 - Sessions: Federated Central pretraining")
            # experiment_pain(algorithm='federated',
            #                 dataset='PAIN',
            #                 experiment='19-sessions-Federated-pre-training',
            #                 rounds=30,
            #                 shards=None,
            #                 clients=None,
            #                 model_path=find_newest_model_path(CENTRAL_PAIN_MODELS, "shard-0.00.h5"),
            #                 pretraining='centralized',
            #                 cumulative=True,
            #                 optimizer=optimizer,
            #                 loss=loss,
            #                 metrics=metrics,
            #                 model_type='CNN'
            #                 )
            # twilio.send_message("Experiment 19 Complete")
            #
            # training_setup(seed)
            # Output.print_experiment("20 - Sessions: Federated RMSProp")
            # experiment_pain(algorithm='federated',
            #                 dataset='PAIN',
            #                 experiment='20-sessions-Federated-pre-training-RMSProp',
            #                 rounds=30,
            #                 shards=None,
            #                 clients=None,
            #                 model_path=find_newest_model_path(CENTRAL_PAIN_MODELS, "shard-0.00.h5"),
            #                 pretraining='centralized',
            #                 cumulative=True,
            #                 optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
            #                 loss=loss,
            #                 metrics=metrics,
            #                 model_type='CNN'
            #                 )
            # twilio.send_message("Experiment 20 Complete")
            #
            # training_setup(seed)
            # Output.print_experiment("21 - Sessions: Federated Local Epochs")
            # experiment_pain(algorithm='federated',
            #                 dataset='PAIN',
            #                 experiment='21-sessions-Federated-pre-training-Local-Epochs',
            #                 rounds=30,
            #                 shards=None,
            #                 clients=None,
            #                 model_path=find_newest_model_path(CENTRAL_PAIN_MODELS, "shard-0.00.h5"),
            #                 pretraining='centralized',
            #                 cumulative=True,
            #                 optimizer=optimizer,
            #                 loss=loss,
            #                 metrics=metrics,
            #                 model_type='CNN',
            #                 local_epochs=5
            #                 )
            # twilio.send_message("Experiment 21 Complete")

            # mA.LESS_PARAMS = True
            #
            # training_setup(seed)
            # Output.print_experiment("22 - Sessions: Centralized Less Parameters")
            # experiment_pain(algorithm='centralized',
            #                 dataset='PAIN',
            #                 experiment='22-sessions-Centralized-pre-training-Less-Parameters',
            #                 rounds=30,
            #                 shards=None,
            #                 clients=None,
            #                 model_path=None,
            #                 pretraining='centralized',
            #                 cumulative=True,
            #                 optimizer=optimizer,
            #                 loss=loss,
            #                 metrics=metrics,
            #                 model_type='CNN',
            #                 )
            # twilio.send_message("Experiment 22 Complete")

            # training_setup(seed)
            # Output.print_experiment("23 - Sessions: Federated Less Parameters")
            # experiment_pain(algorithm='federated',
            #                 dataset='PAIN',
            #                 experiment='23-sessions-Federated-pre-training-Less-Parameters',
            #                 rounds=30,
            #                 shards=None,
            #                 clients=None,
            #                 model_path=find_newest_model_path(CENTRAL_PAIN_MODELS, "shard-0.00.h5"),
            #                 pretraining='centralized',
            #                 cumulative=True,
            #                 optimizer=optimizer,
            #                 loss=loss,
            #                 metrics=metrics,
            #                 model_type='CNN',
            #                 )
            # twilio.send_message("Experiment 23 Complete")
            #
            # training_setup(seed)
            # Output.print_experiment("24 - Sessions: Centralized ResNet")
            # experiment_pain(algorithm='centralized',
            #                 dataset='PAIN',
            #                 experiment='24-sessions-Centralized-pre-training-ResNet',
            #                 rounds=30,
            #                 shards=None,
            #                 clients=None,
            #                 model_path=None,
            #                 pretraining='centralized',
            #                 cumulative=True,
            #                 optimizer=optimizer,
            #                 loss=loss,
            #                 metrics=metrics,
            #                 model_type='CNN',
            #                 )
            twilio.send_message("Experiment 24 Complete")

        twilio.send_message()

    except Exception as e:
        # twilio.send_message("Attention, an error occurred:\n{}".format(e)[:1000])
        traceback.print_tb(e.__traceback__)
        print(e)

    # Notify that training is complete and shut down Google server
    g_monitor.shutdown()


if __name__ == '__main__':
    main(seed=123, unbalanced=False, balanced=False, sessions=True, redistribution=False)
