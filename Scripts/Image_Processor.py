import os
import cv2
import numpy as np

from Scripts.Data_Loader_Functions import get_labels


def load_and_preprocess_image(path):
    """
    Utility function loading an image, converting it into greyscale, and performing histogram equilization

    :param path:                    string, path for an image
    :return:
    """

    img = cv2.imread(path, 0)  # Load image into greyscale
    img = cv2.equalizeHist(img)  # Histogram equilization
    return img


def bulk_process_images(inputpath, outputpath, extension):
    """
    Utility function processing all images in a given directory.

    :param inputpath:               string, input path
    :param outputpath:              string, output path
    :param extension:               string, extension of the images
    :return:
    """

    for dirpath, dirnames, filenames in os.walk(inputpath):
        structure = os.path.join(outputpath, dirpath[len(inputpath) + 1:])
        for file in filenames:
            if file.endswith(extension):
                src = os.path.join(dirpath, file)
                dest = os.path.join(structure, file)
                img = load_and_preprocess_image(src)
                cv2.imwrite(dest, img)


def bulk_augment_images(input_path, output_path, extension, augmentation, label_type, label_threshold=-1):
    for dir_path, dir_names, filenames in os.walk(input_path):
        structure = os.path.join(output_path, dir_path[len(input_path) + 1:])
        for file in filenames:
            if file.endswith(extension):
                src = os.path.join(dir_path, file)
                label = get_labels([src], label_type)[0]
                if label > label_threshold:
                    img = cv2.imread(src, 0)
                    f_name, f_ext = os.path.splitext(file)
                    if augmentation == 'flip':
                        img = np.flip(img, axis=-1)
                        file = f_name + "_flipped" + f_ext
                    elif augmentation == 'original':
                        file = f_name + "_original" + f_ext
                    else:
                        raise ValueError(
                            "Invalid value for 'augmentation'. Value can be 'flip' or 'original', value was: {}".format(augmentation))
                    dest = os.path.join(structure, file)
                    cv2.imwrite(dest, img)
