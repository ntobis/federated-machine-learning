import os


def remove_files(directory):
    """
    Utility function removing all files in a given directory

    :param directory:               string (a path to a directory)

    """
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
