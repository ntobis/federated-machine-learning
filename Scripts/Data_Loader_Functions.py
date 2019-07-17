import os
import pickle

import numpy as np
import tensorflow as tf

from Scripts import Print_Functions as Output


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


def train_test_split(features, labels, test_split=0.25, shuffle=False):
    if shuffle:
        features, labels = unison_shuffled_copies(features, labels)
    split_point = int(len(features) * test_split)
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


def split_by_label(data, labels):
    split_data = [data[labels == label] for label in np.unique(labels).tolist()]
    split_labels = [labels[labels == label] for label in np.unique(labels).tolist()]
    return split_data, split_labels


def allocate_data(num_clients, split_data, split_labels, categories_per_client, data_points_per_category):
    clients = []
    labels = []
    it = 0
    for idx in range(num_clients):
        client = []
        label = []
        for _ in range(categories_per_client):
            choice_arr = np.random.choice(split_data[it % len(split_data)].shape[0], data_points_per_category)
            client.append(split_data[it % len(split_data)][choice_arr, :])
            label.append(split_labels[it % len(split_labels)][choice_arr])
            it += 1
        client = np.concatenate(client)
        label = np.concatenate(label)
        client, label = unison_shuffled_copies(client, label)
        clients.append(client)
        labels.append(label)
    return clients, labels


def load_data(dataset):
    # Load data
    if dataset == "MNIST":
        train_data, train_labels, test_data, test_labels = load_mnist_data()
    else:
        Output.eprint("No data-set named {}. Loading MNIST instead.".format(dataset))
        train_data, train_labels, test_data, test_labels = load_mnist_data()
        dataset = "MNIST"

    train_data = train_data.astype('float32')
    train_labels = train_labels.astype('float32')
    test_data = test_data.astype('float32')
    test_labels = test_labels.astype('float32')
    return train_data, train_labels, test_data, test_labels, dataset


def sort_data(data, labels):
    sort_array = np.argsort(labels)
    data = data[sort_array]
    labels = labels[sort_array]
    return data, labels


def split_train_data(num_of_clients, train_data, train_labels):
    """
    Splits a dataset into a provided number of clients to simulate a "federated" setting

    :param num_of_clients:          integer specifying the number of clients the data should be split into
    :param train_data:              numpy array
    :param train_labels:            numpy array

    :return:
        train_data:                 numpy array (with additional dimension for N clients)
        train_labels:               numpy array (with additional dimension for N clients)
    """

    # Split data into twice as many shards as clients
    train_data = np.array_split(train_data, num_of_clients * 2)
    train_labels = np.array_split(train_labels, num_of_clients * 2)

    # Shuffle shards so that for sorted data, shards with different labels are adjacent
    train = list(zip(train_data, train_labels))
    np.random.shuffle(train)
    train_data, train_labels = zip(*train)

    # Concatenate adjacent shards
    train_data = [np.concatenate(train_data[i:i+2]) for i in range(0, len(train_data), 2)]
    train_labels = [np.concatenate(train_labels[i:i+2]) for i in range(0, len(train_labels), 2)]

    return train_data, train_labels


def split_data_into_clients(clients, split, train_data, train_labels):
    """
    Utility function to split train data and labels into a specified number of clients, in accordance with a specified
    type of split.

    :param clients:                     int, number of clients the data needs to be split into
    :param split:                       string, type of split that should be performed
    :param train_data:                  numpy array, train data
    :param train_labels:                numpy array, train_labels
    :return:
        train_data                      list of numpy arrays, train_data, split into clients
        train_labels                    list of numpy arrays, train_labels, split into clients
    """

    assert len(train_data) == len(train_labels)

    # Split data
    if split.lower() == 'random':
        train_data, train_labels = split_train_data(clients, train_data, train_labels)
    elif split.lower() == 'overlap':
        train_data, train_labels = sort_data(train_data, train_labels)
        train_data, train_labels = split_train_data(clients, train_data, train_labels)
        for idx in range(len(train_data)):
            train_data[idx], train_labels[idx] = unison_shuffled_copies(train_data[idx], train_labels[idx])
    elif split.lower() == 'no_overlap':
        split_data, split_labels = split_by_label(train_data, train_labels)
        train_data, train_labels = allocate_data(clients,
                                                 split_data,
                                                 split_labels,
                                                 categories_per_client=2,
                                                 data_points_per_category=int(
                                                     len(train_data) / (clients * 2)))
    else:
        raise ValueError(
            "Invalid value for 'Split'. Value can be 'random', 'overlap', 'no_overlap', value was: {}".format(split))
    return train_data, train_labels


def tf_load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=0)
    return tf.image.convert_image_dtype(image, tf.float32)


def get_image_paths(root_path):
    image_paths = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        for file in filenames:
            image_paths.append(os.path.join(dirpath, file))
    return image_paths


def get_labels(image_paths, label_type='pain'):
    label_types = {
        'person': 0,
        'session': 1,
        'culture': 2,
        'frame': 3,
        'pain': 4,
    }

    labels = []
    for path in image_paths:
        filename = os.path.basename(path)
        filename, extension = os.path.splitext(filename)
        img_labels = filename.split("_")
        label = int(img_labels[label_types[label_type]])
        labels.append(label)
    return labels


def load_all_images_into_tf_dataset(path):
    img_paths = get_image_paths(path)

    path_ds = tf.data.Dataset.from_tensor_slices(img_paths)
    image_ds = path_ds.map(tf_load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    labels = get_labels(img_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))

    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    return image_label_ds, len(labels)


def prepare_dataset_for_training(ds, batch_size, ds_size):
    # Setting a shuffle buffer size as large as the dataset ensures that the data is
    # completely shuffled.
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=ds_size)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    # `prefetch` lets the dataset fetch batches, in the background while the model is training.
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds
