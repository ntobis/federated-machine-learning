import os
import cv2

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "Data")
RAW_DATA = os.path.join(DATA, "Raw Data")
PREPROCESSED_DATA = os.path.join(DATA, "Preprocessed Data")


def load_and_preprocess_image(path):
    """
    Utility function loading an image, converting it into greyscale, and performing histogram equilization

    :param path:                    string, path for an image
    :return:
    """

    img = cv2.imread(path, 0)  # Load image into greyscale
    img = cv2.equalizeHist(img)  # Histogram equilization
    return img


def mirror_folder_structure(inputpath, outputpath):
    """
    Utility function mirroring the folder structure in one folder into another folder.

    :param inputpath:               string, input path
    :param outputpath:              string, output path
    :return:
    """

    for dirpath, dirnames, filenames in os.walk(inputpath):
        structure = os.path.join(outputpath, dirpath[len(inputpath) + 1:])
        if not os.path.isdir(structure):
            os.mkdir(structure)
        else:
            print("Folder does already exits!")


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


if __name__ == '__main__':
    mirror_folder_structure(RAW_DATA, PREPROCESSED_DATA)
    bulk_process_images(RAW_DATA, PREPROCESSED_DATA, ".jpg")
