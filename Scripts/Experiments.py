import argparse
import os
import sys
import traceback

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


def save_results(dataset, experiment, history, model, folder):
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
    f_name = time.strftime("%Y-%m-%d-%H%M%S") + "_{}_{}".format(dataset, experiment)
    history.to_csv(os.path.join(folder, f_name))


def load_and_prepare_data(path, person, pain):
    enc = OneHotEncoder(sparse=False, categories='auto')
    data, labels = dL.load_pain_data(path)
    labels_ord = labels[:, pain].astype(np.int)
    labels_binary = dL.reduce_pain_label_categories(labels_ord, max_pain=1)
    train_labels_people = labels[:, person].astype(np.int)
    labels_binary = enc.fit_transform(labels_binary.reshape(len(labels_binary), 1))
    return data, labels_binary, train_labels_people, labels


def split_data_into_clients(train_data, train_labels, clients, subjects_per_client, person):
    if clients is None:
        client_arr = np.unique(train_labels[:, person])
        train_data, train_labels_binary, all_labels = \
            dL.split_data_into_clients(len(client_arr), 'person', train_data, train_labels,
                                       train_labels, subjects_per_client=subjects_per_client)

        # If no clients are specified, clients will be separated according to the "person" label
        clients = all_labels
    else:
        train_data, train_labels_binary = dL.split_data_into_clients(clients, 'random', train_data,
                                                                     train_labels)
    return clients, train_data, train_labels_binary


# ---------------------------------------------- End Utility Functions --------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Experiment Runners ---------------------------------------------- #


def run_pretraining(clients, dataset, experiment, local_epochs, loss, metrics, model_path, model_type, optimizer,
                    pretraining, rounds, subjects_per_client):
    if model_path is not None:
        model = tf.keras.models.load_model(model_path)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    elif pretraining is 'centralized':
        model = mA.build_model((215, 215, 1), model_type)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # Prepare labels for training and evaluation
        train_data, train_labels, train_labels_people, raw_labels = load_and_prepare_data(GROUP_1_TRAIN_PATH, person=0,
                                                                                          pain=4)

        # Train
        model = model_runner(pretraining, dataset, experiment + "_shard-0.00", model=model, rounds=rounds,
                             train_data=train_data, train_labels=train_labels, evaluate=False, loss=loss)

    elif pretraining == 'federated':
        # Load data
        train_data, train_labels, train_labels_people, raw_labels = load_and_prepare_data(GROUP_1_TRAIN_PATH, person=0,
                                                                                          pain=4)

        # Split data into clients
        clients, train_data, train_labels = split_data_into_clients(train_data, train_labels, clients,
                                                                    subjects_per_client, person=0)

        # Train
        model = model_runner(pretraining, dataset, experiment + "_shard-0.00", rounds=rounds, train_data=train_data,
                             train_labels=train_labels, evaluate=False, loss=loss, clients=clients,
                             local_epochs=local_epochs, optimizer=optimizer, metrics=metrics, model_type=model_type)

    elif pretraining is None:
        model = mA.build_model((215, 215, 1), model_type)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    else:
        raise ValueError("Invalid Argument. You must either specify a 'model_path' or provide 'centralized' or "
                         "'federated' as arguments for 'pretraining'.")
    return model


def run_shards(algorithm, cumulative, dataset, experiment, local_epochs, loss, metrics, model, model_type, optimizer,
               rounds, shards, subjects_per_client):
    # Load test data
    test_data, test_labels, test_labels_people, raw_labels = load_and_prepare_data(GROUP_2_TEST_PATH, person=0,
                                                                                   pain=4)
    # Load group 2 training data
    train_data, train_labels, train_labels_people, raw_labels = load_and_prepare_data(GROUP_2_TRAIN_PATH, person=0,
                                                                                      pain=4)
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
            data, labels, all_labels = dL.split_data_into_clients(len(client_arr), 'person', data, labels,
                                                                  all_labels=all_labels,
                                                                  subjects_per_client=subjects_per_client)

        model = model_runner(algorithm, dataset, experiment_current, model=model, rounds=rounds, train_data=data,
                             train_labels=labels, test_data=test_data, test_labels=test_labels,
                             people=test_labels_people, loss=loss, clients=all_labels, local_epochs=local_epochs,
                             optimizer=optimizer, metrics=metrics, model_type=model_type)


def run_sessions(algorithm, dataset, experiment, local_epochs, loss, metrics, model, model_type, optimizer, rounds):
    # Prepare df for data generator
    df = dL.create_pain_df(GROUP_2_PATH)
    # Run Sessions
    for session in df['Session'].unique():
        Output.print_session(session)
        experiment_current = experiment + "_shard-{}".format(session)
        model = model_runner(algorithm, dataset, experiment_current, model=model, rounds=rounds, df=df,
                             evaluate=True, loss=loss, session=session, local_epochs=local_epochs,
                             optimizer=optimizer, metrics=metrics, model_type=model_type)


def model_runner(algorithm, dataset, experiment, model=None, rounds=5, train_data=None, train_labels=None,
                 test_data=None,
                 test_labels=None, df=None, people=None, evaluate=True, loss=None, session=False, clients=None,
                 local_epochs=1, participants=None, optimizer=None, metrics=None, model_type='CNN'):
    """
    Sets up a federated CNN that trains on a specified dataset. Saves the results to CSV.

    :param algorithm:
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
                                               df=df, evaluate=evaluate, loss=loss, people=people, session=session,
                                               clients=clients, local_epochs=local_epochs,
                                               participating_clients=participants, optimizer=optimizer, metrics=metrics,
                                               model_type=model_type)

    elif algorithm is 'centralized':
        folder = CENTRAL_PAIN_MODELS
        model, history = cP.train_cnn(model=model, epochs=rounds, train_data=train_data,
                                      train_labels=train_labels, test_data=test_data,
                                      test_labels=test_labels, df=df, people=people, evaluate=evaluate,
                                      loss=loss, session=session)

    else:
        raise ValueError("'runner_type' must be either 'centralized' or 'federated', was: {}".format(algorithm))

    save_results(dataset, experiment, history, model, folder)

    return model


# ---------------------------------------------- End Experiment Runners -------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Experiments - PAIN ---------------------------------------------- #

def experiment_pain(algorithm, dataset, experiment, rounds, shards=None, clients=None, model_path=None,
                    pretraining=None, cumulative=True, optimizer=None, loss=None, metrics=None,
                    subjects_per_client=None, local_epochs=1, model_type='CNN'):

    # Perform pre-training on group 1
    model = run_pretraining(clients, dataset, experiment, local_epochs, loss, metrics, model_path, model_type,
                            optimizer, pretraining, rounds, subjects_per_client)

    # If shards are specified, this experiment will be run
    if shards is not None:
        run_shards(algorithm, cumulative, dataset, experiment, local_epochs, loss, metrics, model, model_type,
                   optimizer, rounds, shards, subjects_per_client)

    # Else, split group 2 into sessions and run this experiment
    else:
        run_sessions(algorithm, dataset, experiment, local_epochs, loss, metrics, model, model_type, optimizer, rounds)


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
                            clients=None, pretraining=None, cumulative=True, optimizer=optimizer, loss=loss,
                            metrics=metrics, subjects_per_client=1)
            twilio.send_message("Experiment 3 Complete")

            # Experiment 4 - Unbalanced: Federated with centralized pretraining
            training_setup(seed)
            Output.print_experiment("4 - Unbalanced: Federated with centralized pretraining")
            centralized_model_path = find_newest_model_path(os.path.join(CENTRAL_PAIN_MODELS),
                                                            "shard-0.00.h5")
            experiment_pain("federated", 'PAIN', '4-unbalanced-Federated-central-pre-training', 30, shards=test_shards,
                            clients=None,
                            model_path=centralized_model_path, pretraining='centralized', cumulative=True,
                            optimizer=optimizer, loss=loss, metrics=metrics, subjects_per_client=1)
            twilio.send_message("Experiment 4 Complete")

            # Experiment 5 - Unbalanced: Federated with federated pretraining
            training_setup(seed)
            Output.print_experiment("5 - Unbalanced: Federated with federated pretraining")
            experiment_pain("federated", 'PAIN', '5-unbalanced-Federated-federated-pre-training', 30,
                            shards=test_shards,
                            clients=None, pretraining='federated', cumulative=True, optimizer=optimizer, loss=loss,
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
                            clients=None,
                            pretraining=None, cumulative=True, optimizer=optimizer, loss=loss, metrics=metrics,
                            subjects_per_client=1)
            twilio.send_message("Experiment 8 Complete")

            # Experiment 9 - Balanced: Federated with centralized pretraining
            training_setup(seed)
            Output.print_experiment("9 - Balanced: Federated with centralized pretraining")
            centralized_model_path = find_newest_model_path(os.path.join(CENTRAL_PAIN_MODELS),
                                                            "shard-0.00.h5")
            experiment_pain("federated", 'PAIN', '4-balanced-Federated-central-pre-training', 30, shards=test_shards,
                            clients=None,
                            model_path=centralized_model_path, pretraining='centralized', cumulative=True,
                            optimizer=optimizer, loss=loss, metrics=metrics, subjects_per_client=1)
            twilio.send_message("Experiment 9 Complete")

            # Experiment 10 - Balanced: Federated with federated pretraining
            training_setup(seed)
            Output.print_experiment("10 - Balanced: Federated with federated pretraining")
            experiment_pain("federated", 'PAIN', '5-balanced-Federated-federated-pre-training', 30, shards=test_shards,
                            clients=None,
                            pretraining='federated', cumulative=True, optimizer=optimizer, loss=loss, metrics=metrics,
                            subjects_per_client=1)
            twilio.send_message("Experiment 10 Complete")

        # --------------------------------------- SESSIONS ---------------------------------------#

        if sessions:
            if redistribution:
                training_setup(seed)
                dL.prepare_pain_images(data_loc, distribution='sessions')
            #
            # # Experiment 11 - Sessions: Centralized without pre-training
            # training_setup(seed)
            # Output.print_experiment("11 - Sessions: Centralized without pre-training")
            # experiment_pain(algorithm='centralized',
            #                 dataset='PAIN',
            #                 experiment='1-sessions-Centralized-no-pre-training',
            #                 rounds=30,
            #                 shards=None,
            #                 clients=None,
            #                 model_path=None,
            #                 pretraining=None,
            #                 cumulative=True,
            #                 optimizer=optimizer,
            #                 loss=loss,
            #                 metrics=metrics,
            #                 model_type='CNN'
            #                 )
            # twilio.send_message("Experiment 11 Complete")
            #
            # # Experiment 12 - Sessions: Centralized with pre-training
            # training_setup(seed)
            # Output.print_experiment("12 - Sessions: Centralized with pre-training")
            # experiment_pain(algorithm='centralized',
            #                 dataset='PAIN',
            #                 experiment='2-sessions-Centralized-pre-training',
            #                 rounds=30,
            #                 shards=None,
            #                 clients=None,
            #                 model_path=None,
            #                 pretraining='centralized',
            #                 cumulative=True,
            #                 optimizer=optimizer,
            #                 loss=loss,
            #                 metrics=metrics,
            #                 model_type='CNN'
            #                 )
            # twilio.send_message("Experiment 12 Complete")

            # Experiment 13 - Sessions: Federated without pre-training
            training_setup(seed)
            Output.print_experiment("13 - Sessions: Federated without pre-training")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='3-sessions-Federated-no-pre-training',
                            rounds=2,
                            shards=None,
                            clients=None,
                            model_path=None,
                            pretraining=None,
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            subjects_per_client=1,
                            local_epochs=1,
                            model_type='ResNet'
                            )
            twilio.send_message("Experiment 13 Complete")

            # Experiment 14 - Sessions: Federated with centralized pretraining
            training_setup(seed)
            Output.print_experiment("14 - Sessions: Federated with centralized pretraining")
            centralized_model_path = find_newest_model_path(CENTRAL_PAIN_MODELS, "shard-0.00.h5")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='4-sessions-Federated-central-pre-training',
                            rounds=30,
                            shards=None,
                            clients=None,
                            model_path=centralized_model_path,
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
                            clients=None,
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

            # Experiment 16 - Sessions: Federated 'BEST' architecture
            training_setup(seed)
            Output.print_experiment("15 - Sessions: Federated with federated pretraining")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='5-sessions-Federated-federated-pre-training',
                            rounds=30,
                            shards=None,
                            clients=None,
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

        twilio.send_message()

    except Exception as e:
        twilio.send_message("Attention, an error occurred:\n{}".format(e)[:1000])
        traceback.print_tb(e.__traceback__)
        print(e)

    # Notify that training is complete and shut down Google server
    # g_monitor.shutdown()


if __name__ == '__main__':
    main(seed=123, unbalanced=False, balanced=False, sessions=True, redistribution=False)
