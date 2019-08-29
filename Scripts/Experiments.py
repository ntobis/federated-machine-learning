import argparse
import os
import sys
import traceback

import pandas as pd
from tensorflow.python.keras.metrics import TruePositives, TrueNegatives, FalsePositives, FalseNegatives, Recall, \
    Precision, AUC

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time

import numpy as np
import tensorflow as tf
from twilio.rest import Client

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

from Scripts import Print_Functions as pF
from Scripts import Data_Loader_Functions as dL
from Scripts import Model_Training as mT
from Scripts import Model_Architectures as mA
from Scripts.Weights_Accountant import WeightsAccountant

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
    def __init__(self, project, zone, instance):
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

    color = 0 if model_type == 'CNN' else 1
    data, labels = dL.load_pain_data(path, color=color)
    labels_ord = labels[:, pain].astype(np.int)
    labels_binary = dL.reduce_pain_label_categories(labels_ord, max_pain=1)
    train_labels_people = labels[:, person].astype(np.int)
    return data, labels_binary, train_labels_people, labels


def evaluate_model(model, test_data, test_labels, test_people, test_all_labels, split_type, split,
                   weights_accountant):
    # Prepare data
    history = {metric: [] for metric in model.metrics_names}

    # Evaluate overall data set
    results = model.evaluate(test_data, test_labels)
    for key, val in zip(model.metrics_names, results):
        history.setdefault(key, []).append(val)
    history.setdefault(split_type, []).append(split)

    # Split data into test subjects
    _, test_data_split, test_labels_split, test_people_split = dL.split_data_into_labels(0, test_all_labels, False,
                                                                                         test_data, test_labels,
                                                                                         test_people)

    # Evaluate each test subject
    for data, labels, people, in zip(test_data_split, test_labels_split, test_people_split):
        if weights_accountant is not None:
            weights_accountant.apply_client_weights(model, people[0])
        results = model.evaluate(data, labels)
        for key, val in zip(model.metrics_names, results):
            history.setdefault('subject_{}_'.format(people[0]) + key, []).append(val)

    history = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in history.items()]))

    return history


def evaluate_session(df_history, df_testing, model, model_type, session, weights_accountant=None):
    # Get test data
    df_test = df_testing[df_testing['Session'] == session]
    test_data, test_labels, test_people, test_all_labels = load_and_prepare_data(
        df_test['img_path'].values,
        person=0,
        pain=4,
        model_type=model_type)

    # Evaluate the model on the test data
    results = evaluate_model(model, test_data, test_labels, test_people, test_all_labels, 'Session', session,
                             weights_accountant)
    df_history = pd.concat((df_history, results), sort=False)
    return df_history


def baseline_model_evaluation(dataset, experiment, model_path, optimizer, loss, metrics, model_type):
    df_history = pd.DataFrame()
    df_testing = dL.create_pain_df(GROUP_2_PATH, pain_gap=())
    model = tf.keras.models.load_model(model_path)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    for session in df_testing['Session'].unique():
        if session > 0:
            pF.print_session(session)
            df_history = evaluate_session(df_history, df_testing, model, model_type, session)

    # Save history to CSV
    f_name = time.strftime("%Y-%m-%d-%H%M%S") + "_{}_{}.csv".format(dataset, experiment + "_TEST")
    df_history.to_csv(os.path.join(RESULTS, f_name))


# ---------------------------------------------- End Utility Functions --------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Experiment Runners ---------------------------------------------- #

def run_pretraining(dataset, experiment, local_epochs, loss, metrics, model_path, model_type, optimizer,
                    pretraining, rounds, pain_gap):
    if model_path is not None:
        print("Loading pre-trained model: {}".format(os.path.basename(model_path)))
        model = tf.keras.models.load_model(model_path)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    elif pretraining is 'centralized':
        print("Pre-training a centralized model.")
        model = mA.build_model((215, 215, 1), model_type)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # Prepare labels for training and evaluation
        df = dL.create_pain_df(GROUP_1_TRAIN_PATH, pain_gap=pain_gap)
        train_data, train_labels, train_people, raw_labels = load_and_prepare_data(df['img_path'].values,
                                                                                   person=0,
                                                                                   pain=4,
                                                                                   model_type=model_type)
        # Train
        model = model_runner(pretraining, dataset, experiment + "_shard-0.00", model=model, rounds=rounds,
                             train_data=train_data, train_labels=train_labels, individual_validation=False)

    elif pretraining == 'federated':
        print("Pre-training a federated model.")
        model = mA.build_model((215, 215, 1), model_type)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        weights_accountant = WeightsAccountant(model)

        # Load data
        df = dL.create_pain_df(GROUP_1_TRAIN_PATH, pain_gap=pain_gap)
        data, labels, people, all_labels = load_and_prepare_data(df['img_path'].values,
                                                                 person=0,
                                                                 pain=4, model_type=model_type)

        # Split data into train and validation
        data, labels, people, all_labels = train_test_split(0.2, data, labels, people, all_labels)
        train_data, val_data = data
        train_labels, val_labels = labels
        train_people, val_people = people
        train_labels_all, val_labels_all = all_labels

        clients = np.unique(train_people)

        # Train
        model = model_runner(pretraining, dataset, experiment + "_shard-0.00", model=model, rounds=rounds,
                             train_data=train_data, train_labels=train_labels, train_people=train_people,
                             val_data=val_data, val_labels=val_labels, val_people=val_people,
                             val_all_labels=val_labels_all, clients=clients, local_epochs=local_epochs,
                             individual_validation=False, weights_accountant=weights_accountant)

    elif pretraining is None:
        model = mA.build_model((215, 215, 1), model_type)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    else:
        raise ValueError("Invalid Argument. You must either specify a 'model_path' or provide 'centralized' or "
                         "'federated' as arguments for 'pretraining'.")
    return model


def train_test_split(test_ratio, *args):
    split = int(len(args[0]) * test_ratio)
    array = [(elem[split:], elem[:split]) for elem in args]
    return tuple(array)


def split_and_balance_df(df, ratio, balance_test=False):
    # Split Original data into ratio
    df_original = df[(df['Trans_1'] == 'original') & (df['Trans_2'] == 'straight')]
    df_train = df_original.sample(frac=1).groupby('Person', group_keys=False).apply(lambda x: x.sample(frac=ratio))
    df_test = df_original.drop(df_train.index)

    # Balance the training data set
    df_train = df[df['temp_id'].isin(df_train['temp_id'])]
    df_pain = df_train[df_train['Pain'] > 0]
    df_train = pd.concat((df_pain, df_train[df_train['Pain'] == 0].sample(len(df_pain))), ignore_index=True)

    if balance_test:
        df_test = df[df['temp_id'].isin(df_test['temp_id'])]
        df_pain = df_test[df_test['Pain'] > 0]
        df_test = pd.concat((df_pain, df_test[df_test['Pain'] == 0].sample(len(df_pain))), ignore_index=True)

    # Return shuffled dfs
    return df_train.sample(frac=1), df_test.sample(frac=1)


def run_shards(algorithm, cumulative, dataset, experiment, local_epochs, model, model_type, rounds, shards,
               pain_gap, individual_validation, local_operation, balance_test):
    # Initialize WeightsAccountant
    weights_accountant = WeightsAccountant(model) if algorithm == 'federated' else None

    # Prepare df for data loading and for history tracking
    df = dL.create_pain_df(GROUP_2_PATH, pain_gap=pain_gap)
    df_train, df_test = split_and_balance_df(df, shards[-1], balance_test)
    df_history = pd.DataFrame()

    # Load test data
    test_data, test_labels, test_people, test_all_labels = load_and_prepare_data(df_test['img_path'].values, person=0,
                                                                                 pain=4, model_type=model_type)
    # Load group 2 training data
    train_data, train_labels, train_people, train_all_labels = load_and_prepare_data(df_train['img_path'].values,
                                                                                     person=0,
                                                                                     pain=4, model_type=model_type)
    # Split group 2 training data into shards
    split_train_data, split_train_labels, split_train_people, split_train_all_labels = dL.split_data_into_shards(
        array=[train_data, train_labels, train_people, train_all_labels], split=shards, cumulative=cumulative)

    # Train on group 2 shards and evaluate performance
    for percentage, train_data, train_labels, train_people, train_all_labels in \
            zip(shards, split_train_data, split_train_labels, split_train_people, split_train_all_labels):
        pF.print_shard(percentage)
        experiment_current = experiment + "_shard-{}".format(percentage)

        # Define clients
        clients = np.unique(train_people)

        # Train
        model = model_runner(algorithm, dataset, experiment_current, model, rounds, train_data, train_labels,
                             train_people, clients=clients, local_epochs=local_epochs,
                             individual_validation=individual_validation, local_operation=local_operation,
                             weights_accountant=weights_accountant)

        # Evaluate
        df_history = evaluate_model(model, test_data, test_labels, test_people, test_all_labels, 'Shard', percentage,
                                    weights_accountant)

    # Save history to CSV
    f_name = time.strftime("%Y-%m-%d-%H%M%S") + "_{}_{}.csv".format(dataset, experiment + "_TEST")
    df_history.to_csv(os.path.join(RESULTS, f_name))


def run_sessions(algorithm, dataset, experiment, local_epochs, model, model_type, rounds, pain_gap,
                 individual_validation, local_operation):
    # Initialize WeightsAccountant
    weights_accountant = WeightsAccountant(model) if algorithm == 'federated' else None

    # Prepare df for data loading and for history tracking
    df_training_validating = dL.create_pain_df(GROUP_2_PATH, pain_gap=pain_gap)
    df_testing = dL.create_pain_df(GROUP_2_PATH, pain_gap=())
    df_history = pd.DataFrame()

    # Run Sessions
    train_data, train_labels, train_people, train_all_labels, client_arr = [None] * 5
    for session in df_training_validating['Session'].unique():
        pF.print_session(session)
        experiment_current = experiment + "_shard-{}".format(session)

        if session > 0:
            clients = np.unique(train_people)

            # Get test-set evaluation data
            df_history = evaluate_session(df_history, df_testing, model, model_type, session, weights_accountant)

            # Get validation data
            df_val = df_training_validating[df_training_validating['Session'] == session]
            val_data, val_labels, val_people, val_all_labels = load_and_prepare_data(
                df_val['img_path'].values,
                person=0,
                pain=4,
                model_type=model_type)

            # Train the model
            model = model_runner(algorithm, dataset, experiment_current, model=model, rounds=rounds,
                                 train_data=train_data, train_labels=train_labels, train_people=train_people,
                                 val_data=val_data, val_labels=val_labels, val_people=val_people,
                                 val_all_labels=val_all_labels, clients=clients, local_epochs=local_epochs,
                                 individual_validation=individual_validation, local_operation=local_operation,
                                 weights_accountant=weights_accountant)

        # Get Train Data for the next session
        df_train = df_training_validating[df_training_validating['Session'] <= session]
        df_train = dL.balance_data(df_train, threshold=200)
        train_data, train_labels, train_people, train_all_labels = load_and_prepare_data(
            df_train['img_path'].values,
            person=0, pain=4, model_type=model_type)

    # Save history to CSV
    f_name = time.strftime("%Y-%m-%d-%H%M%S") + "_{}_{}.csv".format(dataset, experiment + "_TEST")
    df_history.to_csv(os.path.join(RESULTS, f_name))


def model_runner(algorithm, dataset, experiment, model=None, rounds=5, train_data=None, train_labels=None,
                 train_people=None, val_data=None, val_labels=None, val_people=None, val_all_labels=None, clients=None,
                 local_epochs=1, individual_validation=True, local_operation='global_averaging',
                 weights_accountant=None):
    """
    Sets up a federated CNN that trains on a specified dataset. Saves the results to CSV.

    :param train_people:
    :param weights_accountant:
    :param local_operation:
    :param individual_validation:
    :param val_all_labels:
    :param algorithm:
    :param clients:                 int, the maximum number of clients participating in a communication round
    :param dataset:                 string, name of the dataset to be used, e.g. "MNIST"
    :param experiment:              string, the type of experimental setting to be used, e.g. "CLIENTS"
    :param train_data:              numpy array, the train data
    :param train_labels:            numpy array, the train labels
    :param val_data:               numpy array, the test data
    :param val_labels:             numpy array, the test labels
    :param rounds:                  int, number of communication rounds that the federated clients average results for
    :param local_epochs:                  int, number of epochs that the client CNN trains for
    :param val_people:                  numpy array of len test_labels, enabling individual client metrics
    :param model:                   A compiled tensorflow model
    :return:
    """

    if algorithm is 'federated':
        folder = FEDERATED_PAIN_MODELS

        # Train Model
        history, model = mT.federated_learning(model, rounds, train_data, train_labels, train_people, val_data,
                                               val_labels, val_people, val_all_labels, clients, local_epochs,
                                               individual_validation, local_operation, weights_accountant)

    elif algorithm is 'centralized':
        folder = CENTRAL_PAIN_MODELS
        model, history = mT.train_cnn(algorithm,
                                      model,
                                      rounds,
                                      train_data,
                                      train_labels,
                                      val_data,
                                      val_labels,
                                      val_people,
                                      val_all_labels,
                                      individual_validation
                                      )

    else:
        raise ValueError("'runner_type' must be either 'centralized' or 'federated', was: {}".format(algorithm))

    history = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in history.items()]))
    save_results(dataset, experiment, history, model, folder)

    return model


# ---------------------------------------------- End Experiment Runners -------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Experiments - PAIN ---------------------------------------------- #

def experiment_pain(algorithm, dataset, experiment, rounds, shards=None, balance_test_set=False, model_path=None,
                    pretraining=None, cumulative=True, optimizer=None, loss=None, metrics=None, local_epochs=1,
                    model_type='CNN', pain_gap=(), individual_validation=True, local_operation='global_averaging'):
    # Perform pre-training on group 1
    model = run_pretraining(dataset, experiment, local_epochs, loss, metrics, model_path, model_type,
                            optimizer, pretraining, rounds, pain_gap)

    # If shards are specified, this experiment will be run
    if shards is not None:
        run_shards(algorithm, cumulative, dataset, experiment, local_epochs, model, model_type,
                   rounds, shards, pain_gap, individual_validation, local_operation, balance_test_set)

    # Else, split group 2 into sessions and run this experiment
    else:
        run_sessions(algorithm, dataset, experiment, local_epochs, model, model_type, rounds,
                     pain_gap, individual_validation, local_operation)


# ------------------------------------------------ End Experiments - 3 --------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


def main(seed=123, unbalanced=False, balanced=False, sessions=False, evaluate=False):
    # Setup
    twilio = Twilio()

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = ['accuracy', TruePositives(), TrueNegatives(),
               FalsePositives(), FalseNegatives(), Recall(), Precision(), AUC(curve='ROC', name='auc'),
               AUC(curve='PR', name='pr')]

    model_type = 'CNN'
    pain_gap = ()

    test_shards = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    try:

        # --------------------------------------- UNBALANCED ---------------------------------------#
        if unbalanced:

            # Experiment 1 - Unbalanced: Centralized without pre-training
            training_setup(seed)
            pF.print_experiment("1 - Unbalanced: Centralized without pre-training")
            experiment_pain(algorithm='centralized',
                            dataset='PAIN',
                            experiment='1-unbalanced-Centralized-no-pre-training' + "_" + str(seed),
                            rounds=30,
                            shards=test_shards,
                            balance_test_set=False,
                            model_path=None,
                            pretraining=None,
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            model_type='CNN',
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='global_averaging'
                            )
            twilio.send_message("Experiment 1 Complete")

            # Experiment 2 - Unbalanced: Centralized with pre-training
            training_setup(seed)
            pF.print_experiment("2 - Unbalanced: Centralized with pre-training")
            experiment_pain(algorithm='centralized',
                            dataset='PAIN',
                            experiment='2-unbalanced-Centralized-pre-training' + "_" + str(seed),
                            rounds=30,
                            shards=test_shards,
                            balance_test_set=False,
                            model_path=None,
                            pretraining='centralized',
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            model_type='CNN',
                            pain_gap=pain_gap,
                            individual_validation=False
                            )
            twilio.send_message("Experiment 2 Complete")

            # Experiment 3 - Unbalanced: Federated without pre-training
            training_setup(seed)
            pF.print_experiment("3 - Unbalanced: Federated without pre-training")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='3-unbalanced-Federated-no-pre-training' + "_" + str(seed),
                            rounds=30,
                            shards=test_shards,
                            balance_test_set=False,
                            model_path=None,
                            pretraining=None,
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            local_epochs=5,
                            model_type='CNN',
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='global_averaging'
                            )
            twilio.send_message("Experiment 3 Complete")

            # Experiment 4 - Unbalanced: Federated with centralized pretraining
            training_setup(seed)
            pF.print_experiment("4 - Unbalanced: Federated with centralized pretraining")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='4-unbalanced-Federated-central-pre-training' + "_" + str(seed),
                            rounds=30,
                            shards=test_shards,
                            balance_test_set=False,
                            model_path=find_newest_model_path(CENTRAL_PAIN_MODELS, "shard-0.00.h5"),
                            pretraining='centralized',
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            local_epochs=5,
                            model_type='CNN',
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='global_averaging'
                            )
            twilio.send_message("Experiment 4 Complete")

            # Experiment 5 - Unbalanced: Federated with federated pretraining
            training_setup(seed)
            pF.print_experiment("5 - Unbalanced: Federated with federated pretraining")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='5-unbalanced-Federated-federated-pre-training' + "_" + str(seed),
                            rounds=30,
                            shards=test_shards,
                            balance_test_set=False,
                            model_path=None,
                            pretraining='federated',
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            local_epochs=5,
                            model_type='CNN',
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='global_averaging'
                            )
            twilio.send_message("Experiment 5 Complete")

            if evaluate:
                baseline_model_evaluation(dataset="PAIN",
                                          experiment="0-unbalanced-Baseline-central-pre-training" + "_" + str(seed),
                                          model_path=find_newest_model_path(CENTRAL_PAIN_MODELS, "_shard-0.00.h5"),
                                          optimizer=optimizer,
                                          loss=loss,
                                          metrics=metrics,
                                          model_type=model_type
                                          )

                baseline_model_evaluation(dataset="PAIN",
                                          experiment="0-unbalanced-Baseline-federated-pre-training" + "_" + str(seed),
                                          model_path=find_newest_model_path(FEDERATED_PAIN_MODELS, "_shard-0.00.h5"),
                                          optimizer=optimizer,
                                          loss=loss,
                                          metrics=metrics,
                                          model_type=model_type
                                          )

        # --------------------------------------- BALANCED ---------------------------------------#

        if balanced:

            # Experiment 6 - Balanced: Centralized without pre-training
            training_setup(seed)
            pF.print_experiment("6 - Balanced: Centralized without pre-training")
            experiment_pain(algorithm='centralized',
                            dataset='PAIN',
                            experiment='1-balanced-Centralized-no-pre-training' + "_" + str(seed),
                            rounds=30,
                            shards=test_shards,
                            balance_test_set=True,
                            model_path=None,
                            pretraining=None,
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            model_type='CNN',
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='global_averaging'
                            )
            twilio.send_message("Experiment 6 Complete")

            # Experiment 7 - Balanced: Centralized with pre-training
            training_setup(seed)
            pF.print_experiment("7 - Balanced: Centralized with pre-training")
            experiment_pain(algorithm='centralized',
                            dataset='PAIN',
                            experiment='2-balanced-Centralized-pre-training' + "_" + str(seed),
                            rounds=30,
                            shards=test_shards,
                            balance_test_set=True,
                            model_path=None,
                            pretraining='centralized',
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            model_type='CNN',
                            pain_gap=pain_gap,
                            individual_validation=False
                            )
            twilio.send_message("Experiment 7 Complete")

            # Experiment 8 - Balanced: Federated without pre-training
            training_setup(seed)
            pF.print_experiment("8 - Balanced: Federated without pre-training")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='3-balanced-Federated-no-pre-training' + "_" + str(seed),
                            rounds=30,
                            shards=test_shards,
                            balance_test_set=True,
                            model_path=None,
                            pretraining=None,
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            local_epochs=5,
                            model_type='CNN',
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='global_averaging'
                            )
            twilio.send_message("Experiment 8 Complete")

            # Experiment 9 - Balanced: Federated with centralized pretraining
            training_setup(seed)
            pF.print_experiment("9 - Balanced: Federated with centralized pretraining")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='4-balanced-Federated-central-pre-training' + "_" + str(seed),
                            rounds=30,
                            shards=test_shards,
                            balance_test_set=True,
                            model_path=find_newest_model_path(CENTRAL_PAIN_MODELS, "shard-0.00.h5"),
                            pretraining='centralized',
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            local_epochs=5,
                            model_type='CNN',
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='global_averaging'
                            )
            twilio.send_message("Experiment 9 Complete")

            # Experiment 10 - Balanced: Federated with federated pretraining
            training_setup(seed)
            pF.print_experiment("10 - Balanced: Federated with federated pretraining")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='5-balanced-Federated-federated-pre-training' + "_" + str(seed),
                            rounds=30,
                            shards=test_shards,
                            balance_test_set=True,
                            model_path=None,
                            pretraining='federated',
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            local_epochs=5,
                            model_type='CNN',
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='global_averaging'
                            )
            twilio.send_message("Experiment 10 Complete")

            if evaluate:
                baseline_model_evaluation(dataset="PAIN",
                                          experiment="0-balanced-Baseline-central-pre-training" + "_" + str(seed),
                                          model_path=find_newest_model_path(CENTRAL_PAIN_MODELS, "_shard-0.00.h5"),
                                          optimizer=optimizer,
                                          loss=loss,
                                          metrics=metrics,
                                          model_type=model_type
                                          )

                baseline_model_evaluation(dataset="PAIN",
                                          experiment="0-balanced-Baseline-federated-pre-training" + "_" + str(seed),
                                          model_path=find_newest_model_path(FEDERATED_PAIN_MODELS, "_shard-0.00.h5"),
                                          optimizer=optimizer,
                                          loss=loss,
                                          metrics=metrics,
                                          model_type=model_type
                                          )

        # --------------------------------------- SESSIONS ---------------------------------------#

        if sessions:
            # Experiment 11 - Sessions: Centralized without pre-training
            training_setup(seed)
            pF.print_experiment("11 - Sessions: Centralized without pre-training")
            experiment_pain(algorithm='centralized',
                            dataset='PAIN',
                            experiment='1-sessions-Centralized-no-pre-training' + "_" + str(seed),
                            rounds=30,
                            shards=None,
                            model_path=None,
                            pretraining=None,
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            model_type=model_type,
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='global_averaging'
                            )
            twilio.send_message("Experiment 11 Complete")

            # Experiment 12 - Sessions: Centralized with pre-training
            training_setup(seed)
            pF.print_experiment("12 - Sessions: Centralized with pre-training")
            experiment_pain(algorithm='centralized',
                            dataset='PAIN',
                            experiment='2-sessions-Centralized-pre-training' + "_" + str(seed),
                            rounds=30,
                            shards=None,
                            model_path=None,
                            pretraining='centralized',
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            model_type=model_type,
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='global_averaging'
                            )
            twilio.send_message("Experiment 12 Complete")

            # Experiment 13 - Sessions: Federated without pre-training
            training_setup(seed)
            pF.print_experiment("13 - Sessions: Federated without pre-training")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='3-sessions-Federated-no-pre-training' + "_" + str(seed),
                            rounds=30,
                            shards=None,
                            model_path=None,
                            pretraining=None,
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            local_epochs=5,
                            model_type=model_type,
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='global_averaging'
                            )
            twilio.send_message("Experiment 13 Complete")

            # Experiment 14 - Sessions: Federated with centralized pretraining
            training_setup(seed)
            pF.print_experiment("14 - Sessions: Federated with centralized pretraining")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='4-sessions-Federated-central-pre-training' + "_" + str(seed),
                            rounds=30,
                            shards=None,
                            model_path=find_newest_model_path(CENTRAL_PAIN_MODELS, "shard-0.00.h5"),
                            pretraining='centralized',
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            local_epochs=5,
                            model_type=model_type,
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='global_averaging'
                            )
            twilio.send_message("Experiment 14 Complete")

            # Experiment 15 - Sessions: Federated with federated pretraining
            training_setup(seed)
            pF.print_experiment("15 - Sessions: Federated with federated pretraining")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='5-sessions-Federated-federated-pre-training' + "_" + str(seed),
                            rounds=30,
                            shards=None,
                            model_path=None,
                            pretraining='federated',
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            local_epochs=5,
                            model_type=model_type,
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='global_averaging'
                            )
            twilio.send_message("Experiment 15 Complete")

            # Experiment 16 - Sessions: Federated without pre-training
            training_setup(seed)
            pF.print_experiment("16 - Sessions: Federated without pre-training")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='6-sessions-Federated-no-pre-training-personalization' + "_" + str(seed),
                            rounds=30,
                            shards=None,
                            model_path=None,
                            pretraining=None,
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            local_epochs=5,
                            model_type=model_type,
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='localized_learning'
                            )
            twilio.send_message("Experiment 16 Complete")

            # Experiment 17 - Sessions: Federated with centralized pretraining
            training_setup(seed)
            pF.print_experiment("17 - Sessions: Federated with centralized pretraining")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='7-sessions-Federated-central-pre-training-personalization' + "_" + str(seed),
                            rounds=30,
                            shards=None,
                            model_path=find_newest_model_path(CENTRAL_PAIN_MODELS, "shard-0.00.h5"),
                            pretraining='centralized',
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            local_epochs=5,
                            model_type=model_type,
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='localized_learning'
                            )
            twilio.send_message("Experiment 17 Complete")

            # Experiment 18 - Sessions: Federated with federated pretraining
            training_setup(seed)
            pF.print_experiment("18 - Sessions: Federated with federated pretraining")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='8-sessions-Federated-federated-pre-training-personalization' + "_" + str(seed),
                            rounds=30,
                            shards=None,
                            model_path=find_newest_model_path(FEDERATED_PAIN_MODELS, "shard-0.00.h5"),
                            pretraining='federated',
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            local_epochs=5,
                            model_type=model_type,
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='localized_learning'
                            )
            twilio.send_message("Experiment 18 Complete")

            # Experiment 19 - Sessions: Federated without pre-training
            training_setup(seed)
            pF.print_experiment("19 - Sessions: Federated without pre-training")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='9-sessions-Federated-no-pre-training-local-models' + "_" + str(seed),
                            rounds=30,
                            shards=None,
                            model_path=None,
                            pretraining=None,
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            local_epochs=5,
                            model_type=model_type,
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='local_models'
                            )
            twilio.send_message("Experiment 19 Complete")

            # Experiment 20 - Sessions: Federated with centralized pretraining
            training_setup(seed)
            pF.print_experiment("20 - Sessions: Federated with centralized pretraining")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='10-sessions-Federated-central-pre-training-local-models' + "_" + str(seed),
                            rounds=30,
                            shards=None,
                            model_path=find_newest_model_path(CENTRAL_PAIN_MODELS, "shard-0.00.h5"),
                            pretraining='centralized',
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            local_epochs=5,
                            model_type=model_type,
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='local_models'
                            )
            twilio.send_message("Experiment 20 Complete")

            # Experiment 21 - Sessions: Federated with federated pretraining
            training_setup(seed)
            pF.print_experiment("21 - Sessions: Federated with federated pretraining")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='11-sessions-Federated-federated-pre-training-local-models' + "_" + str(seed),
                            rounds=30,
                            shards=None,
                            model_path=find_newest_model_path(FEDERATED_PAIN_MODELS, "shard-0.00.h5"),
                            pretraining='federated',
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            local_epochs=5,
                            model_type=model_type,
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='local_models')
            twilio.send_message("Experiment 21 Complete")

            if evaluate:
                baseline_model_evaluation(dataset="PAIN",
                                          experiment="0-sessions-Baseline-central-pre-training" + "_" + str(seed),
                                          model_path=find_newest_model_path(CENTRAL_PAIN_MODELS, "_shard-0.00.h5"),
                                          optimizer=optimizer,
                                          loss=loss,
                                          metrics=metrics,
                                          model_type=model_type
                                          )

                baseline_model_evaluation(dataset="PAIN",
                                          experiment="0-sessions-Baseline-federated-pre-training" + "_" + str(seed),
                                          model_path=find_newest_model_path(FEDERATED_PAIN_MODELS, "_shard-0.00.h5"),
                                          optimizer=optimizer,
                                          loss=loss,
                                          metrics=metrics,
                                          model_type=model_type
                                          )

                twilio.send_message("Evaluation Complete")

            move_files('{} - Seed {}'.format(seed, seed), seed)

    except Exception as e:
        twilio.send_message("Attention, an error occurred:\n{}".format(e)[:1000])
        traceback.print_tb(e.__traceback__)
        print(e)


def move_files(target_folder, seed):
    # Create folder structure
    target_f_path = os.path.join(RESULTS, 'Thesis', target_folder)
    if not os.path.isdir(target_f_path):
        os.mkdir(target_f_path)

    if not os.path.isdir(os.path.join(target_f_path, 'Plotting')):
        os.mkdir(os.path.join(target_f_path, 'Plotting'))

    # Move files and folders
    elements = [elem for elem in os.listdir(RESULTS) if str(seed) in elem]
    for elem in elements:
        f_path = os.path.join(RESULTS, elem)
        os.rename(f_path, os.path.join(target_f_path, elem))

    # Delete Seed number from file and folder names
    elements = [elem for elem in os.listdir(target_f_path) if "_" + str(seed) in elem]
    for elem in elements:
        f_path = os.path.join(target_f_path, elem)
        new = elem.replace("_" + str(seed), '')
        os.rename(f_path, os.path.join(target_f_path, new))


if __name__ == '__main__':
    # vm = 1
    # inst = 'federated-' + str(vm) + '-vm'
    # g_monitor = GoogleCloudMonitor(project='inbound-column-251110', zone='us-west1-b', instance=inst)
    # main(seed=125, unbalanced=False, balanced=False, sessions=True, evaluate=True)
    # g_monitor.shutdown()
    seed = 130
    move_files('{} - Seed {}'.format(seed, seed), seed)
