import os
import pickle

import numpy as np
import tensorflow as tf

from Scripts import Print_Functions as Output
from Scripts.Centralized_CNN import AUTISM


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def get_pickle_files(directory):
    files = []
    for file in os.listdir(directory):
        if file.endswith(".pkl"):
            files.append(os.path.join(directory, file))
    files = sorted(files)
    return files


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        file = pickle.load(f)
    return file


def prepare_autism_data(file):
    frame_number_col = 5
    splits = 6, 257, 327, 354, 378, 393

    frames = file[frame_number_col]
    offset_file = file[:, splits[0]:]
    face = np.expand_dims(offset_file[:, :splits[1]], axis=0)
    body = np.expand_dims(offset_file[:, splits[1]:splits[2]], axis=0)
    phy = np.expand_dims(offset_file[:, splits[2]:splits[3]], axis=0)
    audio = np.expand_dims(offset_file[:, splits[3]:splits[4]], axis=0)
    cars = np.expand_dims(offset_file[:, splits[4]:splits[5]], axis=0)
    labels = np.expand_dims(offset_file[:, -1:], axis=0)
    return frames, face, body, phy, audio, cars, labels


def load_autism_data_into_clients(folder_path):
    files = get_pickle_files(folder_path)
    clients_frames, clients_face, clients_body, clients_phy, clients_audio, clients_cars, clients_labels = \
        [], [], [], [], [], [], []

    for file_name in files:
        file = load_pickle(file_name)
        frames, face, body, phy, audio, cars, labels = prepare_autism_data(file)
        clients_frames.append(frames)
        clients_face.append(face)
        clients_body.append(body)
        clients_phy.append(phy)
        clients_audio.append(audio)
        clients_cars.append(cars)
        clients_labels.append(labels)
    return clients_frames, clients_face, clients_body, clients_phy, clients_audio, clients_cars, clients_labels


def train_test_split(features, labels, test_split=0.25, shuffle=False):
    if shuffle:
        features, labels = unison_shuffled_copies(features, labels)
    split_point = int(len(features)*test_split)
    train_data = features[split_point:]
    test_data = features[:split_point]
    train_labels = features[split_point:]
    test_labels = features[:split_point]
    return train_data, train_labels, test_data, test_labels


def load_mnist_data():
    """
    Loads the MNIST Data Set and reshapes it for further model training

    :return:
        train_images        numpy array of shape (60000, 28, 28, 1)
        train_labels        numpy array of shape (60000, )
        test_images         numpy array of shape (10000, 28, 28, 1)
        test_labels         numpy array of shape (10000, )
    """

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    return train_images, train_labels, test_images, test_labels


def load_autism_data_body():
    clients = load_autism_data_into_clients(AUTISM)
    return clients[2], clients[-1]


def load_data(dataset):
    # Load data
    if dataset == "MNIST":
        train_data, train_labels, test_data, test_labels = load_mnist_data()
    else:
        Output.eprint("No data-set named {}. Loading MNIST instead.".format(dataset))
        train_data, train_labels, test_data, test_labels = load_mnist_data()
        dataset = "MNIST"
    return train_data, train_labels, test_data, test_labels, dataset