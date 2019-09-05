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
from Scripts.Keras_Custom import focal_loss

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------ Paths ----------------------------------------------------- #

ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS = os.path.join(ROOT, 'Results')

MODELS = os.path.join(ROOT, 'Models')
CENTRAL_PAIN_MODELS = os.path.join(MODELS, "Pain", "Centralized")
FEDERATED_PAIN_MODELS = os.path.join(MODELS, "Pain", "Federated")

DATA = os.path.join(ROOT, "Data", "Augmented Data", "Flexible Augmentation")
GROUP_1_PATH = os.path.join(DATA, "group_1")
GROUP_2_PATH = os.path.join(DATA, "group_2")


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
        if project != '' and zone != '' and instance != '':
            self.credentials = GoogleCredentials.get_application_default()
            self.service = discovery.build('compute', 'v1', credentials=self.credentials)

            # Project ID for this request.
            self.project = project

            # The name of the zone for this request.
            self.zone = zone

            # Name of the instance resource to stop.
            self.instance = instance
            self.initialized = True

        else:
            self.initialized = False

    def shutdown(self):
        """
        API call to shuts down a Google instance.
        :return:
        """

        if self.initialized:
            request = self.service.instances().stop(project=self.project, zone=self.zone, instance=self.instance)
            return request.execute()


class Twilio(Client):
    def __init__(self, account, pw, sender, receiver):
        """
        Instantiate a Twilio Client that sends text messages when training is complete or an error occurs. Parses login
        credentials from the command line.
        Check www.twilio.com for detailed instructions.

        """

        # Parse Commandline Arguments
        self.sender = sender
        self.receiver = receiver
        if account != '' and pw != '' and sender != '' and receiver != '':
            super(Twilio, self).__init__(account, pw)
            self.initialized = True

        else:
            self.initialized = False

    def send_message(self, msg=None):
        """
        Sends a text message.

        :param msg:                 string, message sent. If not specified one of the default messages will be sent.
        :return:
        """

        body = ['Sir, this is Google speaking. Your Federated model trained like a boss. Google out.',
                "Wow, what a ride. Training complete.",
                "This was wild. But I trained like crazy. We're done here."]
        if msg is None:
            msg = np.random.choice(body)
        if self.initialized:
            self.messages.create(to=self.receiver, from_=self.sender, body=msg)


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


def baseline_model_evaluation(dataset, experiment, model_path, optimizer, loss, metrics, model_type):
    """
    Utility function used to evaluate a saved Tensorflow model. If no model path is provided, a random model is
    initialized and evaluated. Calls evaluate_session.


    :param dataset:                 string, name of the dataset used for training
    :param experiment:              string, name of the experiment conducted
    :param model_path:              string, absolute file path to a tensorflow .h5 file
    :param optimizer:               Instantiated Tensorflow Optimizer Object
    :param loss:                    Instantiated Tensorflow Loss Object
    :param metrics:                 list, list of instantiated Tensorflow metric objects
    :param model_type:              string, 'CNN' or 'ResNet'
    :return:
    """
    # Create history object (data frame)
    df_history = pd.DataFrame()

    # Load all file paths of GROUP 2 into a dataframe
    df_testing = dL.create_pain_df(GROUP_2_PATH, pain_gap=())

    # Instantiate and compile a Tensorflow model
    if model_path is None:
        model = mA.build_model((215, 215, 1), model_type)
    else:
        model = tf.keras.models.load_model(model_path, custom_objects={'focal_loss_function': focal_loss()})
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Evaluate the model for all sessions
    for session in df_testing['Session'].unique():
        if session > 0:
            pF.print_session(session)
            df_history = evaluate_session(df_history, df_testing, model, model_type, session)

    # Save history to CSV
    f_name = time.strftime("%Y-%m-%d-%H%M%S") + "_{}_{}.csv".format(dataset, experiment + "_TEST")
    df_history.to_csv(os.path.join(RESULTS, f_name))


def evaluate_session(df_history, df_testing, model, model_type, session, weights_accountant=None):
    """
    Utility function preparing all images from a given session to be evaluated. Calls evaluate_model and returns a data
    frame containing the evaluation history.

    :param df_history:                  Pandas DataFrame, stores the evaluation history
    :param df_testing:                  Pandas DataFrame, cols: [Person, Session, Culture, Frame, Pain, Trans_1,
                                                                 Trans_2, img_path]
    :param model:                       Tensorflow Graph
    :param model_type:                  string, 'CNN' or 'ResNet'
    :param session:                     int, session to be evaluated
    :param weights_accountant:          WeightsAccountant object, must be provided for federated model evaluation

    :return:
        df_history, concatenated with updated results

    """
    # Get test data
    df_test = df_testing[df_testing['Session'] == session]
    test_data, test_labels, test_people, test_all_labels = dL.load_and_prepare_pain_data(
        df_test['img_path'].values,
        person=0,
        pain=4,
        model_type=model_type)

    # Evaluate the model on the test data
    results = evaluate_model(model, test_data, test_labels, test_people, test_all_labels, 'Session', session,
                             weights_accountant)

    return pd.concat((df_history, results), sort=False)


def evaluate_model(model, test_data, test_labels, test_people, test_all_labels, split_type, split, weights_accountant):
    """
    Utility function called to evaluate a model. Upgrades the Tensorflow evaluate() function by adding functionality for
    federated evaluation. Adds None, when a given client was not evaluated, to maintain consistent records across
    epochs.

    :param model:                       Tensorflow Graph
    :param test_data:                   numpy array, contains image data, (set_size, img_height, img_width, channels)
    :param test_labels:                 numpy array, contains image labels (set_size, 1)
    :param test_people:                 numpy array, contains image clients (set_size, 1)
    :param test_all_labels:             numpy array, contains all labels obtained from .jpg file (set_size, len(labels))
    :param split_type:                  string, typically 'Session' or 'Shard', but can be anything
    :param split:                       float or int, specifying the session number / shard
    :param weights_accountant:          WeightsAccountant Object, necessary for federated evaluation, else can be None

    :return:
         Pandas DataFrame, containing the training history
    """
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

    return pd.DataFrame(dict([(k, pd.Series(v)) for k, v in history.items()]))


# ---------------------------------------------- End Utility Functions --------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Experiment Runners ---------------------------------------------- #

def run_pretraining(dataset, experiment, local_epochs, optimizer, loss, metrics, model_path, model_type, pretraining,
                    rounds, pain_gap):
    """
    Runs the pre-training. If a model path is specified the model is loaded. Else, if pretraining is "centralized",
    a new model is initialized and pre-trained on GROUP 1 in a centralized manner. Else, if pretraining is "federated" a
    new model is initialized and pre-trained on GROUP 1 in a federated manner. Else if pre-training is None, a random
    model is initialized.

    :param dataset:                 string, name of the dataset used for training
    :param experiment:              string, name of the experiment conducted
    :param local_epochs:            int, number of local epochs to run in a federated setting
    :param optimizer:               Instantiated Tensorflow Optimizer Object
    :param loss:                    Instantiated Tensorflow Loss Object
    :param metrics:                 list, list of instantiated Tensorflow metric objects
    :param model_path:              string, file path to .h5 file
    :param model_type:              string, 'CNN' or 'ResNet'
    :param pretraining:             string, 'centralized' or 'federated' or None
    :param rounds:                  int, global total number of communication rounds or epochs
    :param pain_gap:                tuple of int's, specifying which pain classes to exclude from training

    :return:
        Compiled TensorFlow graph
    """

    # Load existing model
    if model_path is not None:
        print("Loading pre-trained model: {}".format(os.path.basename(model_path)))
        model = tf.keras.models.load_model(model_path, custom_objects={'focal_loss_function': focal_loss()})
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Pre-train in centralized fashion
    elif pretraining is 'centralized':
        print("Pre-training a centralized model.")
        model = mA.build_model((215, 215, 1), model_type)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # Prepare labels for training and evaluation
        df = dL.create_pain_df(GROUP_1_PATH, pain_gap=pain_gap)
        df, _ = dL.split_and_balance_df(df, ratio=1, balance_test=False)
        train_data, train_labels, _, _ = dL.load_and_prepare_pain_data(df['img_path'].values,
                                                                       person=0,
                                                                       pain=4,
                                                                       model_type=model_type)

        # Train
        model = model_runner(pretraining, dataset, experiment + "_shard-0.00", model=model, rounds=rounds,
                             train_data=train_data, train_labels=train_labels, individual_validation=False)

    # Pre-train in federated fashion
    elif pretraining == 'federated':
        print("Pre-training a federated model.")
        model = mA.build_model((215, 215, 1), model_type)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        weights_accountant = WeightsAccountant(model)

        # Load data
        df = dL.create_pain_df(GROUP_1_PATH, pain_gap=pain_gap)
        df, _ = dL.split_and_balance_df(df, ratio=1, balance_test=False)
        data, labels, people, all_labels = dL.load_and_prepare_pain_data(df['img_path'].values,
                                                                         person=0,
                                                                         pain=4, model_type=model_type)

        # Split data into train and validation
        data, labels, people, all_labels = dL.train_test_split(0.8, data, labels, people, all_labels)
        train_data, val_data = data
        train_labels, val_labels = labels
        train_people, val_people = people
        train_labels_all, val_labels_all = all_labels

        # Get clients
        clients = np.unique(train_people)

        # Train
        model = model_runner(pretraining, dataset, experiment + "_shard-0.00", model=model, rounds=rounds,
                             train_data=train_data, train_labels=train_labels, train_people=train_people,
                             val_data=val_data, val_labels=val_labels, val_people=val_people,
                             val_all_labels=val_labels_all, clients=clients, local_epochs=local_epochs,
                             individual_validation=False, weights_accountant=weights_accountant)

    # Do not pre-train
    elif pretraining is None:
        model = mA.build_model((215, 215, 1), model_type)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    else:
        raise ValueError("Invalid Argument. You must either specify a 'model_path' or provide 'centralized' or "
                         "'federated' as arguments for 'pretraining'.")
    return model


def run_shards(algorithm, cumulative, dataset, experiment, local_epochs, model, model_type, rounds, shards,
               pain_gap, individual_validation, local_operation, balance_test):
    """
    Runs the experimental setting with data split into randomized shards.

    :param algorithm:               string, either 'centralized' or 'federated'
    :param cumulative:              bool, if true, shard data will be added cumulatively, else each shard will be
                                    trained on separately
    :param dataset:                 string, name of the dataset used for training
    :param experiment:              string, name of the experiment conducted
    :param local_epochs:            int, number of local epochs to run in a federated setting
    :param model:                   TensorFlow Graph
    :param model_type:              string, 'CNN' or 'ResNet'
    :param rounds:                  int, global total number of communication rounds or epochs
    :param shards:                  list, list of floats containing the shard percentages e.g. [0.01, 0.1, 0.3, 0.6]
    :param pain_gap:                tuple of int's, specifying which pain classes to exclude from training
    :param individual_validation:   bool, if true, validation history for every local epoch in a federated setting is
                                    stored (typically not necessary)
    :param local_operation:         string, operation to be performed by the WeightsAccountant, valid entries are:
                                    'global_averaging', 'localized_learning' and 'local_models'
    :param balance_test:            bool, whether to balance the test set during evaluation

    :return:
    """

    # Initialize WeightsAccountant
    weights_accountant = WeightsAccountant(model) if algorithm == 'federated' else None

    # Prepare df for data loading and for history tracking
    df = dL.create_pain_df(GROUP_2_PATH, pain_gap=pain_gap)
    df_train, df_test = dL.split_and_balance_df(df, shards[-1], balance_test)
    df_history = pd.DataFrame()

    # Load test data
    test_data, test_labels, test_people, test_all_labels = dL.load_and_prepare_pain_data(df_test['img_path'].values,
                                                                                         person=0,
                                                                                         pain=4, model_type=model_type)
    # Load group 2 training data
    train_data, train_labels, train_people, train_all_labels = dL.load_and_prepare_pain_data(
        df_train['img_path'].values,
        person=0,
        pain=4, model_type=model_type)

    # Split group 2 training data into shards
    split_train_data, split_train_labels, split_train_people, split_train_all_labels = dL.split_data_into_shards(
        array=[train_data, train_labels, train_people, train_all_labels], split=shards, cumulative=cumulative)

    # Train on group 2 shards and evaluate performance
    for percentage, data, labels, people, all_labels in \
            zip(shards, split_train_data, split_train_labels, split_train_people, split_train_all_labels):
        pF.print_shard(percentage)
        experiment_current = experiment + "_shard-{}".format(percentage)

        # Define clients
        clients = np.unique(train_people)

        # Split data into train and validation
        (train_data, val_data), (train_labels, val_labels), (train_people, val_people), (train_all_labels,
                                                                                         val_all_labels) = \
            dL.train_test_split(0.8, data, labels, people, all_labels)

        # Train
        model = model_runner(algorithm, dataset, experiment_current, model, rounds, train_data, train_labels,
                             train_people, val_data=val_data, val_labels=val_labels, val_people=val_people,
                             val_all_labels=val_all_labels, clients=clients, local_epochs=local_epochs,
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
    """
    Run the experimental setting with data split into sessions.

    :param algorithm:               string, either 'centralized' or 'federated'
    :param dataset:                 string, name of the dataset used for training
    :param experiment:              string, name of the experiment conducted
    :param local_epochs:            int, number of local epochs to run in a federated setting
    :param model:                   TensorFlow Graph
    :param model_type:              string, 'CNN' or 'ResNet'
    :param rounds:                  int, global total number of communication rounds or epochs
    :param pain_gap:                tuple of int's, specifying which pain classes to exclude from training
    :param individual_validation:   bool, if true, validation history for every local epoch in a federated setting is
                                    stored (typically not necessary)
    :param local_operation:         string, operation to be performed by the WeightsAccountant, valid entries are:
                                    'global_averaging', 'localized_learning' and 'local_models'
    :return:
    """

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

        # Only start training on session 0, when session 1 can serve as a test / validation set
        if session > 0:
            clients = np.unique(train_people)

            # Evaluate test set
            df_history = evaluate_session(df_history, df_testing, model, model_type, session, weights_accountant)

            # Load validation set
            df_val = df_training_validating[df_training_validating['Session'] == session]
            val_data, val_labels, val_people, val_all_labels = dL.load_and_prepare_pain_data(
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

        # Load Train Data for the next session
        df_train = df_training_validating[df_training_validating['Session'] <= session]
        df_train = dL.balance_data(df_train, threshold=200)
        train_data, train_labels, train_people, train_all_labels = dL.load_and_prepare_pain_data(
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
    Kicks of the training process. Trainig is either centralized or federated. Saves the results to CSV, and the trained
    model to .h5.

    :param algorithm:               string, either 'centralized' or 'federated'
    :param dataset:                 string, name of the dataset used for training
    :param experiment:              string, name of the experiment conducted
    :param model:                   TensorFlow Graph
    :param rounds:                  int, global total number of communication rounds or epochs
    :param train_data:              numpy array, contains image data, (set_size, img_height, img_width, channels)
    :param train_labels:            numpy array, contains image labels (set_size, 1)
    :param train_people:            numpy array, contains image clients (set_size, 1)
    :param val_data:                numpy array, contains image data, (set_size, img_height, img_width, channels)
    :param val_labels:              numpy array, contains image labels (set_size, 1)
    :param val_people:              numpy array, contains image clients (set_size, 1)
    :param val_all_labels:          numpy array, contains all labels obtained from .jpg file (set_size, len(labels))
    :param clients:                 numpy array, unique client indeces, e.g. [43, 59, 120]
    :param local_epochs:            int, number of local epochs to run in a federated setting
    :param individual_validation:   bool, if true, validation history for every local epoch in a federated setting is
                                    stored (typically not necessary)
    :param local_operation:         string, operation to be performed by the WeightsAccountant, valid entries are:
                                    'global_averaging', 'localized_learning' and 'local_models'
    :param weights_accountant:      WeightsAccountant object

    :return:
        TensorFlow Graph
    """

    if algorithm is 'federated':
        save_model_folder = FEDERATED_PAIN_MODELS

        # Train Model
        history, model = mT.federated_learning(model, rounds, train_data, train_labels, train_people, val_data,
                                               val_labels, val_people, val_all_labels, clients, local_epochs,
                                               individual_validation, local_operation, weights_accountant)

    elif algorithm is 'centralized':
        save_model_folder = CENTRAL_PAIN_MODELS
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
        raise ValueError("'algorithm' must be either 'centralized' or 'federated', was: {}".format(algorithm))

    history = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in history.items()]))
    save_results(dataset, experiment, history, model, save_model_folder)

    return model


# ---------------------------------------------- End Experiment Runners -------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Experiments - PAIN ---------------------------------------------- #

def experiment_pain(algorithm='centralized', dataset='PAIN', experiment='placeholder', setting=None, rounds=30,
                    shards=None, balance_test_set=False, model_path=None, pretraining=None, cumulative=True,
                    optimizer=None, loss=None, metrics=None, local_epochs=1, model_type='CNN', pain_gap=(),
                    individual_validation=True, local_operation='global_averaging'):
    """
    Experiment launcher. Takes all arguments that can be freely tuned to generate new experimental settings. Runs
    pre-training if specified, and either the sessions or the shards experimental setting.

    :param algorithm:               string, either 'centralized' or 'federated'
    :param dataset:                 string, name of the dataset used for training
    :param experiment:              string, name of the experiment conducted
    :param setting:                 string, either 'shards' or 'sessions', specifying the experimental setting
    :param rounds:                  int, global total number of communication rounds or epochs
    :param shards:                  list, REQUIRED for the 'shards' experimental setting. list of floats containing the
                                    shard percentages e.g. [0.01, 0.1, 0.3, 0.6].
    :param balance_test_set:        bool, whether to balance the test set during evaluation. Only has an effect for the
                                    "shards" experimental setting
    :param model_path:              string, file path to .h5 file. If given, pre-trained model will be loaded.
    :param pretraining:             string, 'centralized' or 'federated' or None
    :param cumulative:              bool, if true, shard data will be added cumulatively, else each shard will be
                                    trained on separately. Has no impact on 'sessions'
    :param optimizer:               Instantiated Tensorflow Optimizer Object
    :param loss:                    Instantiated Tensorflow Loss Object
    :param metrics:                 list, list of instantiated Tensorflow metric objects
    :param local_epochs:            int, number of local epochs to run in a federated setting. Has no impact for
                                    centralized learning.
    :param model_type:              string, 'CNN' or 'ResNet'
    :param pain_gap:                tuple of int's, specifying which pain classes to exclude from training, e.g. (1)
    :param individual_validation:   bool, if true, validation history for every local epoch in a federated setting is
                                    stored (typically not necessary)
    :param local_operation:         string, operation to be performed by the WeightsAccountant, valid entries are:
                                    'global_averaging', 'localized_learning' and 'local_models'
    :return:
    """
    # Perform pre-training on group 1
    model = run_pretraining(dataset, experiment, local_epochs, optimizer, loss, metrics, model_path, model_type,
                            pretraining, rounds, pain_gap)

    # If shards are specified, this experiment will be run
    if setting.lower() == 'shards':
        run_shards(algorithm, cumulative, dataset, experiment, local_epochs, model, model_type,
                   rounds, shards, pain_gap, individual_validation, local_operation, balance_test_set)

    # Else, split group 2 into sessions and run this experiment
    elif setting.lower() == 'sessions':
        run_sessions(algorithm, dataset, experiment, local_epochs, model, model_type, rounds,
                     pain_gap, individual_validation, local_operation)
    else:
        print("No training. Valid values for 'setting' are 'shards' and 'sessions'. {} was given".format(setting))

# ------------------------------------------------ End Experiments - 3 --------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


def main(seed=123, shards_unbalanced=False, shards_balanced=False, sessions=False, evaluate=False,
         dest_folder_name='', args=None):
    """
    Main function calling all experimental settings.
    args are helpful if the code is run on a remote server, They can be used to:
    (1) instantiate a Twilio client that sends text messages, once an experiment is complete OR an error occurred, with
        the text message containing an informative error message
    (2) Automatically shut down a Google Cloud Platform (GCP) vm-instance, if this is where the code is run.

    :param seed:                    int, random seed to be set before each experiment
    :param shards_unbalanced:       bool, whether or not to run "randomized shards with unbalanced test set" experiment
    :param shards_balanced:         bool, whether or not to run "randomized shards with balanced test set" experiment
    :param sessions:                bool, whether or not to run "sessions" experiment
    :param evaluate:                bool, whether or not to evaluate random, centralized and federated baseline models.
                                    Requires at least one pre-trained centralized model (.h5 file) in the
                                    CENTRAL_PAIN_MODELS folder and one pre-trained federated model (.h5 file) in the
                                    FEDERATED_PAIN_MODELS folder
    :param dest_folder_name:        string, name of the folder that all results will be moved into in the directory
                                    RESULTS/Thesis
    :param args:
        args.sms_acc                string, Twilio account hash, if None, no messages will be sent
        args.sms_pw                 string, Twilio account password hash, if None, no messages will be sent
        args.sender                 string, Twilio sender number, format "+44XXX", if None, no messages will be sent
        args.receiver               string, Twilio receiver number, format "+44XXX", if None, no messages will be sent
        args.instance               string, GCP instance
        args.project                string, GCP project name
        args.zone                   string, GCP zone

    :return: 
    """
    # Setup SMS Client
    twilio = Twilio(args.sms_acc, args.sms_pw, args.sender, args.receiver)

    # Setup Google Cloud API
    g_monitor = GoogleCloudMonitor(project=args.project, zone=args.zone, instance=args.instance)

    # Tensorflow model compilation objects
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = ['accuracy', TruePositives(), TrueNegatives(),
               FalsePositives(), FalseNegatives(), Recall(), Precision(), AUC(curve='ROC', name='auc'),
               AUC(curve='PR', name='pr')]

    # Model type 'CNN' or 'ResNet'
    model_type = 'CNN'

    # Pain gap, tuple of integers, specifying which pain classes to exclude
    pain_gap = ()

    # Shards to split data into
    test_shards = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    # Send text message
    twilio.send_message("Seed {}".format(seed))
    try:

        # --------------------------------------- UNBALANCED ---------------------------------------#
        if shards_unbalanced:

            # Experiment 1 - Unbalanced: Centralized without pre-training
            training_setup(seed)
            pF.print_experiment("1 - Unbalanced: Centralized without pre-training")
            experiment_pain(algorithm='centralized',
                            dataset='PAIN',
                            experiment='1-unbalanced-Centralized-no-pre-training' + "_" + str(seed),
                            setting='shards',
                            rounds=30,
                            shards=test_shards,
                            balance_test_set=False,
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
            twilio.send_message("Experiment 1 Complete")

            # Experiment 2 - Unbalanced: Centralized with pre-training
            training_setup(seed)
            pF.print_experiment("2 - Unbalanced: Centralized with pre-training")
            experiment_pain(algorithm='centralized',
                            dataset='PAIN',
                            experiment='2-unbalanced-Centralized-pre-training' + "_" + str(seed),
                            setting='shards',
                            rounds=30,
                            shards=test_shards,
                            balance_test_set=False,
                            model_path=None,
                            pretraining='centralized',
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            model_type=model_type,
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
                            setting='shards',
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
                            model_type=model_type,
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
                            setting='shards',
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
                            model_type=model_type,
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
                            setting='shards',
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
                            model_type=model_type,
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='global_averaging'
                            )
            twilio.send_message("Experiment 5 Complete")

            # Experiment 6 - Sessions: Federated without pre-training
            training_setup(seed)
            pF.print_experiment("6 - Sessions: Federated without pre-training")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='6-unbalanced-Federated-no-pre-training-personalization' + "_" + str(seed),
                            setting='shards',
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
                            model_type=model_type,
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='localized_learning'
                            )
            twilio.send_message("Experiment 6 Complete")

            # Experiment 7 - Sessions: Federated with centralized pretraining
            training_setup(seed)
            pF.print_experiment("7 - Sessions: Federated with centralized pretraining")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='7-unbalanced-Federated-central-pre-training-personalization' + "_" + str(seed),
                            setting='shards',
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
                            model_type=model_type,
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='localized_learning'
                            )
            twilio.send_message("Experiment 7 Complete")

            # Experiment 8 - Sessions: Federated with federated pretraining
            training_setup(seed)
            pF.print_experiment("8 - Sessions: Federated with federated pretraining")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='8-unbalanced-Federated-federated-pre-training-personalization' + "_" + str(
                                seed),
                            setting='shards',
                            rounds=30,
                            shards=test_shards,
                            balance_test_set=False,
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
            twilio.send_message("Experiment 8 Complete")

            # Experiment 9 - Sessions: Federated without pre-training
            training_setup(seed)
            pF.print_experiment("9 - Sessions: Federated without pre-training")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='9-unbalanced-Federated-no-pre-training-local-models' + "_" + str(seed),
                            setting='shards',
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
                            model_type=model_type,
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='local_models'
                            )
            twilio.send_message("Experiment 9 Complete")

            # Experiment 10 - Sessions: Federated with centralized pretraining
            training_setup(seed)
            pF.print_experiment("10 - Sessions: Federated with centralized pretraining")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='10-unbalanced-Federated-central-pre-training-local-models' + "_" + str(seed),
                            setting='shards',
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
                            model_type=model_type,
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='local_models'
                            )
            twilio.send_message("Experiment 10 Complete")

            # Experiment 11 - Sessions: Federated with federated pretraining
            training_setup(seed)
            pF.print_experiment("11 - Sessions: Federated with federated pretraining")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='11-unbalanced-Federated-federated-pre-training-local-models' + "_" + str(seed),
                            setting='shards',
                            rounds=30,
                            shards=test_shards,
                            balance_test_set=False,
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
            twilio.send_message("Experiment 11 Complete")

        # --------------------------------------- BALANCED ---------------------------------------#

        if shards_balanced:

            # Experiment 1 - Unbalanced: Centralized without pre-training
            training_setup(seed)
            pF.print_experiment("1 - Unbalanced: Centralized without pre-training")
            experiment_pain(algorithm='centralized',
                            dataset='PAIN',
                            experiment='1-unbalanced-Centralized-no-pre-training' + "_" + str(seed),
                            setting='shards',
                            rounds=30,
                            shards=test_shards,
                            balance_test_set=True,
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
            twilio.send_message("Experiment 1 Complete")

            # Experiment 2 - Unbalanced: Centralized with pre-training
            training_setup(seed)
            pF.print_experiment("2 - Unbalanced: Centralized with pre-training")
            experiment_pain(algorithm='centralized',
                            dataset='PAIN',
                            experiment='2-unbalanced-Centralized-pre-training' + "_" + str(seed),
                            setting='shards',
                            rounds=30,
                            shards=test_shards,
                            balance_test_set=True,
                            model_path=None,
                            pretraining='centralized',
                            cumulative=True,
                            optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            model_type=model_type,
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
                            setting='shards',
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
                            model_type=model_type,
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
                            setting='shards',
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
                            model_type=model_type,
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
                            setting='shards',
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
                            model_type=model_type,
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='global_averaging'
                            )
            twilio.send_message("Experiment 5 Complete")

            # Experiment 6 - Sessions: Federated without pre-training
            training_setup(seed)
            pF.print_experiment("6 - Sessions: Federated without pre-training")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='6-unbalanced-Federated-no-pre-training-personalization' + "_" + str(seed),
                            setting='shards',
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
                            model_type=model_type,
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='localized_learning'
                            )
            twilio.send_message("Experiment 6 Complete")

            # Experiment 7 - Sessions: Federated with centralized pretraining
            training_setup(seed)
            pF.print_experiment("7 - Sessions: Federated with centralized pretraining")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='7-unbalanced-Federated-central-pre-training-personalization' + "_" + str(
                                seed),
                            setting='shards',
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
                            model_type=model_type,
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='localized_learning'
                            )
            twilio.send_message("Experiment 7 Complete")

            # Experiment 8 - Sessions: Federated with federated pretraining
            training_setup(seed)
            pF.print_experiment("8 - Sessions: Federated with federated pretraining")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='8-unbalanced-Federated-federated-pre-training-personalization' + "_" + str(
                                seed),
                            setting='shards',
                            rounds=30,
                            shards=test_shards,
                            balance_test_set=True,
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
            twilio.send_message("Experiment 8 Complete")

            # Experiment 9 - Sessions: Federated without pre-training
            training_setup(seed)
            pF.print_experiment("9 - Sessions: Federated without pre-training")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='9-unbalanced-Federated-no-pre-training-local-models' + "_" + str(seed),
                            setting='shards',
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
                            model_type=model_type,
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='local_models'
                            )
            twilio.send_message("Experiment 9 Complete")

            # Experiment 10 - Sessions: Federated with centralized pretraining
            training_setup(seed)
            pF.print_experiment("10 - Sessions: Federated with centralized pretraining")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='10-unbalanced-Federated-central-pre-training-local-models' + "_" + str(
                                seed),
                            setting='shards',
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
                            model_type=model_type,
                            pain_gap=pain_gap,
                            individual_validation=False,
                            local_operation='local_models'
                            )
            twilio.send_message("Experiment 10 Complete")

            # Experiment 11 - Sessions: Federated with federated pretraining
            training_setup(seed)
            pF.print_experiment("11 - Sessions: Federated with federated pretraining")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='11-unbalanced-Federated-federated-pre-training-local-models' + "_" + str(
                                seed),
                            setting='shards',
                            rounds=30,
                            shards=test_shards,
                            balance_test_set=True,
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
            twilio.send_message("Experiment 11 Complete")

        # --------------------------------------- SESSIONS ---------------------------------------#

        if sessions:
            # Experiment 1 - Sessions: Centralized without pre-training
            training_setup(seed)
            pF.print_experiment("1 - Sessions: Centralized without pre-training")
            experiment_pain(algorithm='centralized',
                            dataset='PAIN',
                            experiment='1-sessions-Centralized-no-pre-training' + "_" + str(seed),
                            setting='sessions',
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
            twilio.send_message("Experiment 1 Complete")

            # Experiment 2 - Sessions: Centralized with pre-training
            training_setup(seed)
            pF.print_experiment("2 - Sessions: Centralized with pre-training")
            experiment_pain(algorithm='centralized',
                            dataset='PAIN',
                            experiment='2-sessions-Centralized-pre-training' + "_" + str(seed),
                            setting='sessions',
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
            twilio.send_message("Experiment 2 Complete")

            # Experiment 3 - Sessions: Federated without pre-training
            training_setup(seed)
            pF.print_experiment("3 - Sessions: Federated without pre-training")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='3-sessions-Federated-no-pre-training' + "_" + str(seed),
                            setting='sessions',
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
            twilio.send_message("Experiment 3 Complete")

            # Experiment 4 - Sessions: Federated with centralized pretraining
            training_setup(seed)
            pF.print_experiment("4 - Sessions: Federated with centralized pretraining")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='4-sessions-Federated-central-pre-training' + "_" + str(seed),
                            setting='sessions',
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
            twilio.send_message("Experiment 4 Complete")

            # Experiment 5 - Sessions: Federated with federated pretraining
            training_setup(seed)
            pF.print_experiment("5 - Sessions: Federated with federated pretraining")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='5-sessions-Federated-federated-pre-training' + "_" + str(seed),
                            setting='sessions',
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
            twilio.send_message("Experiment 5 Complete")

            # Experiment 6 - Sessions: Federated without pre-training
            training_setup(seed)
            pF.print_experiment("6 - Sessions: Federated without pre-training")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='6-sessions-Federated-no-pre-training-personalization' + "_" + str(seed),
                            setting='sessions',
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
            twilio.send_message("Experiment 6 Complete")

            # Experiment 7 - Sessions: Federated with centralized pretraining
            training_setup(seed)
            pF.print_experiment("7 - Sessions: Federated with centralized pretraining")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='7-sessions-Federated-central-pre-training-personalization' + "_" + str(seed),
                            setting='sessions',
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
            twilio.send_message("Experiment 7 Complete")

            # Experiment 8 - Sessions: Federated with federated pretraining
            training_setup(seed)
            pF.print_experiment("8 - Sessions: Federated with federated pretraining")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='8-sessions-Federated-federated-pre-training-personalization' + "_" + str(seed),
                            setting='sessions',
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
            twilio.send_message("Experiment 8 Complete")

            # Experiment 9 - Sessions: Federated without pre-training
            training_setup(seed)
            pF.print_experiment("9 - Sessions: Federated without pre-training")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='9-sessions-Federated-no-pre-training-local-models' + "_" + str(seed),
                            setting='sessions',
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
            twilio.send_message("Experiment 9 Complete")

            # Experiment 10 - Sessions: Federated with centralized pretraining
            training_setup(seed)
            pF.print_experiment("10 - Sessions: Federated with centralized pretraining")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='10-sessions-Federated-central-pre-training-local-models' + "_" + str(seed),
                            setting='sessions',
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
            twilio.send_message("Experiment 10 Complete")

            # Experiment 11 - Sessions: Federated with federated pretraining
            training_setup(seed)
            pF.print_experiment("11 - Sessions: Federated with federated pretraining")
            experiment_pain(algorithm="federated",
                            dataset='PAIN',
                            experiment='11-sessions-Federated-federated-pre-training-local-models' + "_" + str(seed),
                            setting='sessions',
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
            twilio.send_message("Experiment 11 Complete")

        if evaluate:
            baseline_model_evaluation(dataset="PAIN",
                                      experiment="0-sessions-Baseline-random" + "_" + str(seed),
                                      model_path=None,
                                      optimizer=optimizer,
                                      loss=loss,
                                      metrics=metrics,
                                      model_type=model_type
                                      )

            baseline_model_evaluation(dataset="PAIN",
                                      experiment="0-sessions-centralized-pre-training" + "_" + str(seed),
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

        # Move all results into this folder
        dL.move_files(dest_folder_name, seed)

    except Exception as e:
        twilio.send_message("Attention, an error occurred:\n{}".format(e)[:1000])
        traceback.print_tb(e.__traceback__)
        print(e)

    twilio.send_message()

    if g_monitor is not None:
        g_monitor.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sms_acc", help="Enter Twilio Account Here", default='')
    parser.add_argument("--sms_pw", help="Enter Twilio Password Here", default='')
    parser.add_argument("--sender", help="Sender Number", default='')
    parser.add_argument("--receiver", help="Sender Number", default='')
    parser.add_argument("--instance", help="GCP instance that the program runs on.", default='')
    parser.add_argument("--project", help="GCP project that the program runs on.", default='')
    parser.add_argument("--zone", help="GCP zone that the program runs on.", default='')
    parser.add_argument("--seed", help="Random Seed", default=123)
    arguments = parser.parse_args()

    main(seed=int(arguments.seed), shards_unbalanced=False, shards_balanced=False, sessions=True, evaluate=True,
         dest_folder_name='{} - Sd {} - Sessions'.format(int(arguments.seed), int(arguments.seed)), args=arguments)
