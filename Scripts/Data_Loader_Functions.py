import os

import cv2
import numpy as np
import pandas as pd

from Scripts.Experiments import RESULTS


# ------------------------------------------------------------------------------------------------------------------ #
# -------------------------------------------------- Load Pain Data ------------------------------------------------ #

def load_and_prepare_pain_data(path, person, pain, model_type):
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
    data, labels = load_pain_data(path, color=color)
    labels_ord = labels[:, pain].astype(np.int)
    labels_binary = reduce_pain_label_categories(labels_ord, max_pain=1)
    train_labels_people = labels[:, person].astype(np.int)
    return data, labels_binary, train_labels_people, labels


def load_pain_data(train_path, test_path=None, label_type=None, color=0):
    """
    Load function, loading pain dataset into numpy arrays with labels. Either just loads train data, or train and test.

    :param color:               int, if 0 then greyscale, if 1 then color
    :param train_path:          string, root directory
    :param test_path:           string, optional, root test directory
    :param label_type:          string, if not None, only a specific label will be attached to each image
    :return:
    """
    train_data, train_labels = load_image_data(train_path, color, label_type)
    print("Normalization")
    np.divide(train_data, 255.0, out=train_data, dtype=np.float32)
    if test_path:
        test_data, test_labels = load_image_data(test_path, label_type)
        print("Normalization")
        test_data = np.divide(test_data, 255.0, out=test_data, dtype=np.float32)
        return train_data, train_labels, test_data, test_labels
    return train_data, train_labels


def load_image_data(path, color=0, label_type=None):
    """
    Utility function, loading all images in a directory and its sub-directories into a numpy array, labeled

    :param color:               int, if 0 then greyscale, if 1 then color
    :param path:                string, root directory or list of strings (image paths)
    :param label_type:          string, if not None, only a specific label will be attached to each image
    :return:
        data, labels:           tuple of numpy arrays, holding images and labels
    """

    if type(path) is str:
        img_paths = get_image_paths(path)
    else:
        img_paths = path
    np.random.shuffle(img_paths)
    data = []
    for idx, path in enumerate(img_paths):
        img = np.expand_dims(cv2.imread(path, color), -1) if color == 0 else cv2.imread(path, color)
        data.append(img)
        if not idx % 1000:
            print("{} images processed".format(idx))
    data = np.array(data, dtype=np.float32)
    labels = np.array(get_labels(img_paths, label_type=label_type))
    return data, labels


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


def reduce_pain_label_categories(labels, max_pain):
    """
    Utility function reducing ordinal labels to a specified number of labels, e.g. [0,1] binary
    :param labels:              numpy array
    :param max_pain:            int, max label, e.g. if 1 then labels will reduce to [0,1]
    :return:
        labels_reduced          numpy array
    """
    return np.minimum(labels, max_pain)


def create_pain_df(path, pain_gap=(), binarize=True):
    """
    Generate a Pandas DataFrame object that contains all img_paths excluding a specified pain_gap in a given folder path

    :param path:                string, super parent folder path
    :param pain_gap:            tuple of int's, specifying which pain classes to exclude from training
    :param binarize:            bool, if the label "pain" is to be binarized
    :return:
    """

    # Get image paths and convert file labels to numpy array
    img_paths = np.array(get_image_paths(path))
    labels = np.array(get_labels(img_paths))

    # Create dataframe
    df = pd.DataFrame(labels, columns=['Person', 'Session', 'Culture', 'Frame', 'Pain', 'Trans_1', 'Trans_2'])
    df[['Person', 'Session', 'Culture', 'Frame', 'Pain']] = df[
        ['Person', 'Session', 'Culture', 'Frame', 'Pain']].astype(int)
    df['img_path'] = img_paths
    df[['Trans_1', 'Trans_2', 'img_path']] = df[['Trans_1', 'Trans_2', 'img_path']].astype(str)
    df = df.sort_values(['Person', 'Session', 'Frame', 'Trans_1', 'Trans_2'],
                        ascending=[True, True, True, False, False]).reset_index(drop=True)

    # Create a unique ID for each entry
    df['temp_id'] = df['Person'].astype(str) + df['Session'].astype(str) + df['Frame'].astype(str)

    # Exclude elements in the pain gap
    df = df[~df['Pain'].isin(pain_gap)]

    # Binarize Pain label
    if binarize:
        df['Pain'] = np.minimum(df['Pain'], 1)
    return df

# ------------------------------------------------ End Load Pain Data ---------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------- Split Functions ------------------------------------------------ #


def split_data_into_clients_dict(people, *args):
    """
    Utility function, splitting data into clients.

    :param people:              numpy array, contains image clients (set_size, 1)
    :param args:                tuple or list, tuple or list of numpy arrays of the same length as "people"

    :return:
        array of dictionaries. each dictionary represents one input array provided by *args, split into a dictionary of
        clients, where the key is the client number and the value is the data of the input array associated with that
        client
    """
    array = []
    for arg in args:
        dic = {}
        for key, value in zip(people, arg):
            # noinspection PyTypeChecker
            dic.setdefault(key, []).append(value)
        for key in dic.keys():
            dic[key] = np.array(dic[key])
        if len(args) == 1:
            return dic
        else:
            array.append(dic)
    return tuple(array)


def split_data_into_shards(split=None, cumulative=True, array=None):
    """
    Utility function, splitting data into specified subsets of shards. Scales the split array to 100%.

    :param array:               list of arrays to be split into shards
    :param split:               list of percentage split points, e.g. [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    :param cumulative:          bool, checks if concatenation should be cumulative, e.g. [0], [0, 1] or not [0], [1]

    :return:
        list of elements, where each element represents one shard
    """
    split = [int(x / max(split) * len(array[0])) for x in split]
    array = [np.array_split(elem, split) for elem in array]
    if cumulative:
        array = [cumconc(elem) for elem in array]
    return array


def split_data_into_labels(label, all_labels, cumulative, *args):
    """
    Utility function splitting arguments provided in *args into the label provided.

    :param label:               int, labels can be 0-6 each for one index of
                                [person, session, culture, frame, pain, trans_1, trans_2]
    :param all_labels:          2D numpy array, where each row is of the format above
    :param cumulative:          bool, checks if concatenation should be cumulative, e.g. [0], [0, 1] or not [0], [1]
    :param args:                list, arrays to be split into the label provided

    :return:
        tuple of arrays split into the label provided
    """
    arg_array = [np.array([all_labels[all_labels[:, label] == k] for k in np.unique(all_labels[:, label])])]
    arrays = [np.array([arg[all_labels[:, label] == k] for k in np.unique(all_labels[:, label])]) for arg in args]
    arg_array.extend(arrays)
    if cumulative:
        arg_array = [cumconc(elem) for elem in arg_array]
    return tuple(arg_array)


def train_test_split(test_ratio, *args):
    """
    Ordinary train/test split.

    :param test_ratio:          float, split value e.g. 0.8 means 80% train and 20% test data
    :param args:                tuple of arrays to be split

    :return:
        list of split arrays
    """
    split = int(len(args[0]) * test_ratio)
    return [(elem[:split], elem[split:]) for elem in args]


def cumconc(array):
    """
    Utility function creating a cumulatively split view on a split numpy array, e.g. [[0], [1]] to [[0] , [0, 1]]

    :param array:               numpy array
    :return:
        array                   cumulative numpy array
    """
    total = np.concatenate(array)
    # noinspection PyTypeChecker
    return np.array([*map(total.__getitem__, map(slice, np.fromiter(map(len, array), int, len(array)).cumsum()))])


# ----------------------------------------------- End Split Functions ---------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------------------------ #
# ----------------------------------------- Jupyter notebook helper functions -------------------------------------- #

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


def reset_to_raw(root_path, dest_dir='raw', ext='.jpg'):
    """
    Utility function taking all files of a given type in a specified root folder and moving them to a specified
    destination folder.

    :param root_path:           string, root_path
    :param dest_dir:            string, destination folder name
    :param ext:                 string, file extension to look for
    :return:
    """
    if not os.path.isdir(os.path.join(root_path, dest_dir)):
        os.mkdir(os.path.join(root_path, dest_dir))
    for dir_path, dir_names, filenames in os.walk(root_path):
        for file in filenames:
            if os.path.splitext(file)[1] == ext:
                src = os.path.join(dir_path, file)
                dest = os.path.join(root_path, dest_dir, file)
                os.rename(src, dest)


def delete_empty_folders(root_path):
    """
    Utility function deleting all empty folder in a directory and its subdirectories.

    :param root_path:           string, root path
    :return:
    """
    for dir_path, dir_names, filenames in os.walk(root_path):
        if not dir_names and not filenames:
            os.rmdir(dir_path)


def create_pivot(path, index, columns, values, pain_level=0, pain_gap=()):
    """
    Create a pivot table showing the test subjects and their pain level per session.

    :param path:                string, file paths, where the images lie
    :param index:               index of the pivot, can be 'Session' or 'Person'
    :param columns:             columns of the pivot, can be 'Session' or 'Person'
    :param values:              values of the pivot, can be 'Session' or 'Person', should equal index
    :param pain_level:          value from where the binary classifier should consider "pain"
    :param pain_gap:            tuple of int's, specifying which pain classes to exclude from training

    :return:
        Pandas DataFrame, pivot table
    """
    group_2 = get_image_paths(path)
    labels = np.array(get_labels(group_2))
    cols = ['Person', 'Session', 'Culture', 'Frame', 'Pain', 'Trans_1', 'Trans_2']
    df = pd.DataFrame(labels, columns=cols)
    df[['Person', 'Session', 'Culture', 'Frame', 'Pain']] = df[
        ['Person', 'Session', 'Culture', 'Frame', 'Pain']].astype(int)

    df = df[~df['Pain'].isin(pain_gap)]

    pivot = ~df[['Person', 'Session']].drop_duplicates().pivot(index=index, columns=columns, values=values).isnull() * 1
    pivot['# of ' + columns + 's'] = pivot.sum(1)
    pivot = pivot.sort_values('# of ' + columns + 's', ascending=False)

    pivot['Pain'] = 0
    pivot['No Pain'] = 0
    for person, df_person in df.groupby(index):
        pivot.at[person, 'No Pain'] = sum(np.array(df_person['Pain'] <= pain_level))
        pivot.at[person, 'Pain'] = sum(np.array(df_person['Pain'] > pain_level))
        for col in pivot.columns:
            if type(col) is int:
                pivot.at[person, col] = sum(np.array(df_person[df_person[columns] == col]['Pain'] > pain_level))

    if columns is 'Session':
        for col in reversed(pivot.columns):
            if type(col) is int:
                pivot.rename(columns={col: col}, inplace=True)
    if index is 'Session':
        for idx in reversed(pivot.index):
            pivot.rename(index={idx: idx}, inplace=True)
    pivot = pivot.append(pivot.sum(0).rename("Total"))
    pivot['Pain %'] = round(pivot['Pain'] / (pivot['Pain'] + pivot['No Pain']), 2)
    pivot[pivot == 0] = ''
    return pivot


# --------------------------------------- End Jupyter notebook helper functions ------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------------------------ #
# --------------------------------------------- Data Balancing algorithms ------------------------------------------ #

def sample_df(df, threshold):
    """
    Utility function that samples rows from a DataFrame until a provided threshold is reached.

    :param df:                      DataFrame with columns
                                    ['Person', 'Session', 'Culture', 'Frame', 'Pain', 'Trans_1', 'Trans_2', 'temp_id']
    :param threshold:               int, threshold that should be sampled to
    :return:
        Pandas DataFrame
    """
    if len(df) > threshold:
        return df.sample(threshold, replace=False)
    else:
        return pd.concat((df, df.sample(threshold - len(df), replace=True)))


def balance_session(df, threshold):
    """
    Utility function balancing a session so taht equal number of positive and negative examples are included. Only
    includes a client if there are both positive and negative examples for that client

    :param df:                  DataFrame with columns
                                ['Person', 'Session', 'Culture', 'Frame', 'Pain', 'Trans_1', 'Trans_2', 'temp_id']
    :param threshold:           int, threshold that should be sampled to
    :return:
        Resampled Pandas DataFrame or empty DataFrame
    """
    df_train = []
    for person, df_person in df.groupby('Person'):
        df_pain = df_person[df_person['Pain'] == 1]
        df_no_pain = df_person[df_person['Pain'] == 0]
        if len(df_pain) > 0 and len(df_no_pain):
            df_pain = sample_df(df_pain, threshold)
            df_no_pain = sample_df(df_no_pain, threshold)
            df_train.append(pd.concat((df_pain, df_no_pain)))
    return pd.concat(df_train) if len(df_train) > 0 else pd.DataFrame(columns=df.columns)


def balance_data(df, threshold):
    """
    A moving window over the pain data, taking preference over more recent additions to the data set, resampling so
    that an equal number of positive and negative examples is sampled.

    :param df:                  DataFrame with columns
                                ['Person', 'Session', 'Culture', 'Frame', 'Pain', 'Trans_1', 'Trans_2', 'temp_id']
    :param threshold:           int, threshold that should be sampled to
    :return:
        Resampled Pandas DataFrame
    """
    df_train = []
    for person, df_person in df.groupby('Person'):
        df_temp_pain = []
        df_temp_no_pain = []
        for sess, df_sess in reversed(tuple(df_person.groupby('Session'))):
            df_temp_pain.append(df_sess[df_sess['Pain'] == 1])
            df_temp_no_pain.append(df_sess[df_sess['Pain'] == 0])
            if len(pd.concat(df_temp_pain)) > threshold and len(pd.concat(df_temp_no_pain)) > threshold:
                break
        df_temp_pain.extend(df_temp_no_pain)
        df_temp = pd.concat(df_temp_pain)
        df_train.append(df_temp)
    df_train = pd.concat(df_train)
    return balance_session(df_train, threshold)


def split_and_balance_df(df, ratio, balance_test=False):
    """
    Utility function splitting a data frame into train and test file paths. The train data is balanced, while balancing
    the test data is optional. If ratio == 1, serves to just balance a data frame, without splitting the data.

    :param df:                  Pandas DataFrame, cols: [Person, Session, Culture, Frame, Pain, Trans_1, Trans_2,
                                                         img_path]
    :param ratio:               float, ratio of train data
    :param balance_test:        bool, whether to balance the test data

    :return:
        Tuple of two Pandas DataFrames, one with train img_paths, one with test img_paths
    """
    # Split Original data into ratio (for each person)
    df_original = df[(df['Trans_1'] == 'original') & (df['Trans_2'] == 'straight')]
    df_train = df_original.sample(frac=1).groupby('Person', group_keys=False).apply(lambda x: x.sample(frac=ratio))
    df_test = df_original.drop(df_train.index)

    # Balance the training data set (1. get all permutations, 2. get all pain instances of  the permutations,
    # 3. down-sample no-pain to pain number
    df_train = df[df['temp_id'].isin(df_train['temp_id'])]
    df_pain = df_train[df_train['Pain'] > 0]
    df_train = pd.concat((df_pain, df_train[df_train['Pain'] == 0].sample(len(df_pain))), ignore_index=True)

    if balance_test:
        df_test = df[df['temp_id'].isin(df_test['temp_id'])]
        df_pain = df_test[df_test['Pain'] > 0]
        df_test = pd.concat((df_pain, df_test[df_test['Pain'] == 0].sample(len(df_pain))), ignore_index=True)

    # Return shuffled dfs
    return df_train.sample(frac=1), df_test.sample(frac=1)

# ------------------------------------------- End Data Balancing algorithms ---------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


def move_files(target_folder, seed):
    """
    Utility function moving result files into the correct folder.

    :param target_folder:           string, folder where files should be moved to
    :param seed:                    int, seed that was run for the experiments, relevant for file renaming
    :return:
    """
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
