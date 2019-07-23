import os
import pickle

import cv2
import numpy as np
import tensorflow as tf

from Scripts import Print_Functions as Output


def unison_shuffled_copies(a, b):
    """
    Utility function shuffling two numpy arrays in unison.

    :param a:               numpy array
    :param b:               numpy array
    :return:
        a, b                tuple of shuffled numpy arrays
    """

    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def get_pickle_files(directory):
    """
    Utility function returning all pickle file paths in a directory.
    :param directory:       string, directory path
    :return:
        files               sorted list of file paths to pickle files
    """

    files = []
    for file in os.listdir(directory):
        if file.endswith(".pkl"):
            files.append(os.path.join(directory, file))
    files = sorted(files)
    return files


def load_pickle(file_name):
    """
    Utility function loading a pickle file.
    :param file_name:       string, file path
    :return:
        file                pickle file in read mode
    """
    with open(file_name, 'rb') as f:
        file = pickle.load(f)
    return file


def train_test_split(features, labels, test_split=0.25, shuffle=False):
    """
    Utility function splitting features and labels into train and test sets, with the option to shuffle.

    :param features:        numpy array
    :param labels:          numpy array
    :param test_split:      float, indicating the test portion of the data
    :param shuffle:         bool
    :return:
        train_data, train_labels, test_data, test_labels    tuple of numpy arrays
    """
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
    """
    Utility function splitting a data set into as many shards as there are labels, by label
    :param data:            numpy array
    :param labels:          numpy array
    :return:
        split_data, split_labels:   tuple of numpy arrays, split into n-shards
    """

    split_data = [data[labels == label] for label in np.unique(labels).tolist()]
    split_labels = [labels[labels == label] for label in np.unique(labels).tolist()]
    return split_data, split_labels


def allocate_data(num_clients, split_data, split_labels, categories_per_client, data_points_per_category):
    """
    Utility function splitting taking a data set that has been split by label and allocates this data set to a certain
    number of clients. It is possible to specify how many labels a client holds, and how many data point per label a
    client holds.

    :param num_clients:                     int, number of clients
    :param split_data:                      numpy array, split and sorted by label
    :param split_labels:                    numpy array, split and sorted by label
    :param categories_per_client:           int, number of unique labels a client holds
    :param data_points_per_category:        int, number of data point per label a client holds
    :return:
        clients:                            list of numpy arrays, each representing data for one client
        labels:                             list of numpy arrays, each representing labels for one client
    """
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
    """
    Generic data load function.

    :param dataset:                         string, name of dataset
    :return:
        train_data:                         numpy array
        train_labels:                       numpy array
        test_data:                          numpy array
        test_labels:                        numpy array
        dataset:                            string, name of dataset
    """

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
    """
    Utility function sorting a dataset by labels.

    :param data:                            numpy array
    :param labels:                          numpy array
    :return:
        data, labels:                       tuple of sorted numpy arrays
    """

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
    train_data = [np.concatenate(train_data[i:i + 2]) for i in range(0, len(train_data), 2)]
    train_labels = [np.concatenate(train_labels[i:i + 2]) for i in range(0, len(train_labels), 2)]

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
    """
    Load an image into a Tensor.

    :param path:                         string, filepath
    :return:
    Tensor                               Tensorflow Tensor containing image data
    """

    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=0)
    return tf.image.convert_image_dtype(image, tf.float32)


def get_image_paths(root_path, ext='.jpg'):
    """
    Utility function returning all image paths in a directory adn its sub-directories.

    :param root_path:                   path from which to start the recursive search
    :param ext:                         file extension to look for
    :return:
        image_paths:                    list of paths
    """
    image_paths = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        for file in filenames:
            f_name, f_ext = os.path.splitext(file)
            if f_ext == ext:
                image_paths.append(os.path.join(dirpath, file))
    return image_paths


def get_labels(image_paths, label_type=None, ext='.jpg'):
    """
    Utility function turning image paths into a 2D list of labels

    :param image_paths:                 list of image paths
    :param label_type:                  string, if not None, only a specific label per image will be returned
    :param ext:                         string, file extension
    :return:
        labels                          2D list of labels
    """

    label_types = {
        'person': 0,
        'session': 1,
        'culture': 2,
        'frame': 3,
        'pain': 4,
        'augmented': 5,
    }

    labels = []
    for path in image_paths:
        filename = os.path.basename(path)
        filename, extension = os.path.splitext(filename)
        if extension == ext:
            img_labels = filename.split("_")

            if label_type is None:
                label = img_labels
            else:
                label = int(img_labels[label_types[label_type]])

            labels.append(label)
    return labels


def load_all_images_into_tf_dataset(path):
    """
    Utility function loading images in a directory into a Tensorflow Dataset.

    :param path:                    string, root directory path
    :return:
        image_label_ds, len(labels) tuple of Tensorflow dataset and number of data points
    """

    img_paths = get_image_paths(path)

    path_ds = tf.data.Dataset.from_tensor_slices(img_paths)
    image_ds = path_ds.map(tf_load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    labels = get_labels(img_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))

    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    return image_label_ds, len(labels)


def prepare_dataset_for_training(ds, batch_size, ds_size):
    """
    Utility function preparing a Tensorflow Dataset for training.

    :param ds:                  Tensorflow Dataset
    :param batch_size:          int
    :param ds_size:             int, setting the number of images to be cached

    :return:
        ds                      Tensorflow Dataset
    """
    # Setting a shuffle buffer size as large as the dataset ensures that the data is
    # completely shuffled.
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=ds_size)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    # `prefetch` lets the dataset fetch batches, in the background while the model is training.
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def load_greyscale_image_data(path, label_type=None):
    """
    Utility function, loading all images in a directory and its sub-directories into a numpy array, labeled

    :param path:                string, root directory
    :param label_type:          string, if not None, only a specific label will be attached to each image
    :return:
        data, labels:           tuple of numpy arrays, holding images and labels
    """

    img_paths = get_image_paths(path)
    np.random.shuffle(img_paths)
    data = []
    for idx, path in enumerate(img_paths):
        data.append(np.expand_dims(cv2.imread(path, 0), -1))
        if not idx % 1000:
            print("{} images processed".format(idx))
    data = np.array([np.expand_dims(cv2.imread(path, 0), -1) for path in img_paths])
    labels = np.array(get_labels(img_paths, label_type=label_type))
    return data, labels


def load_pain_data(train_path, test_path=None, label_type=None):
    """
    Load function, loading pain dataset into numpy arrays with labels. Either just loads train data, or train and test.

    :param train_path:          string, root directory
    :param test_path:           string, optional, root test directory
    :param label_type:          string, if not None, only a specific label will be attached to each image
    :return:
    """

    train_data, train_labels = load_greyscale_image_data(train_path, label_type)
    train_data = np.divide(train_data, 255.0, dtype=np.float16)
    if test_path:
        test_data, test_labels = load_greyscale_image_data(test_path, label_type)
        test_data = np.divide(test_data, 255.0, dtype=np.float16)
        return train_data, train_labels, test_data, test_labels
    return train_data, train_labels


def reduce_pain_label_categories(labels, max_pain):
    """
    Utility function reducing ordinal labels to a specified number of labels, e.g. [0,1] binary
    :param labels:              numpy array
    :param max_pain:            int, max label, e.g. if 1 then labels will reduce to [0,1]
    :return:
        labels_reduced          numpy array
    """
    return np.minimum(labels, max_pain)


def mirror_folder_structure(input_path, output_path):
    """
    Utility function mirroring the folder structure in one folder into another folder.

    :param input_path:               string, input path
    :param output_path:              string, output path
    :return:
    """

    for dir_path, dir_names, filenames in os.walk(input_path):
        structure = os.path.join(output_path, dir_path[len(input_path) + 1:])
        if not os.path.isdir(structure):
            os.mkdir(structure)


def downsample_data(path):
    """
    Utility function downsampling the number of "no pain" images to the number of "pain images". "No Pain" images are
    randomly deleted.

    :param path:                    string, root filepath
    :return:
    """

    img_paths = np.array(get_image_paths(path))
    img_labels = np.array(get_labels(img_paths))
    pain = img_paths[img_labels[:, 4] != str(0)]
    zero_pain = img_paths[img_labels[:, 4] == str(0)]
    np.random.shuffle(zero_pain)
    delete = zero_pain[:len(zero_pain) - len(pain)]
    [os.remove(file) for file in delete]


def print_pain_label_dist(path):
    """
    Utility function, printing the distribution of "Pain" to "no Pain" data.

    :param path:                    string, root filepath
    :return:
    """

    train_subjects = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    for subject in train_subjects:
        img_paths = get_image_paths(os.path.join(path, subject))
        img_labels = np.array(get_labels(img_paths))
        print("Subject:", subject)
        print("Pain 0:", sum(img_labels[:, 4].astype(int) < 1))
        print("Pain 0<:", sum(img_labels[:, 4].astype(int) >= 1))
        print()


def print_pain_split_per_client(labels):
    """
    Utility function, printing per test subject how many pain/no pain images there are

    :param labels:                  list, list of labels, where the index 4 is the pain label
    :return:
    """

    pain_labels = labels[labels[:, 4].astype(int) > 0]
    no_pain_labels = labels[labels[:, 4].astype(int) == 0]
    pain_unique, pain_count = np.unique(pain_labels[:, 0], return_counts=True)
    no_pain_unique, no_pain_count = np.unique(no_pain_labels[:, 0], return_counts=True)
    for idx, client in enumerate(no_pain_unique):
        for idx_p, client_p in enumerate(pain_unique):
            if client_p == client:
                print("Client:", client, "| Pain / No Pain: {:.2f}".format(pain_count[idx_p] / no_pain_count[idx]))
                break
            if idx_p == len(pain_unique) - 1:
                print("Client:", client, "| Pain / No Pain: {:.2f}".format(0 / no_pain_count[idx]))


def move_train_test_data(df, origin_path, train_path, test_path):
    """
    Randomly samples a Pandas DataFrame into a 40/60 split for each "Person". Then allocates the 40% to a "Test" folder,
    and the 60% to a "Train" folder. To be used with the "Pain Data Preparation Notebook".

    :param df:                      Pandas Dataframe. Must have columns "Person" and "img_paths"
    :param origin_path:             string, path where images lie
    :param train_path:              string, path where the train images go
    :param test_path:               string, path where the test images go
    :return:
    """

    if not os.path.isdir(train_path):
        os.mkdir(train_path)
    if not os.path.isdir(test_path):
        os.mkdir(test_path)
    mirror_folder_structure(origin_path, train_path)
    mirror_folder_structure(origin_path, test_path)

    for person in df['Person'].unique():
        df_client = df[df['Person'] == person]
        train = df_client.sample(frac=0.6, random_state=123)
        test = df_client.drop(train.index)
        for path in train['img_path']:
            folder = os.path.basename(os.path.dirname(path))
            file = os.path.basename(path)
            os.rename(path, os.path.join(train_path, folder, file))
        for path in test['img_path']:
            folder = os.path.basename(os.path.dirname(path))
            file = os.path.basename(path)
            os.rename(path, os.path.join(test_path, folder, file))
