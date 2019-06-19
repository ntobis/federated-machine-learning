import os
import Scripts.Centralized_CNN as cNN
import Scripts.Federated_CNN as fedCNN


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


def main(weights=False, fed_model=False, central_model=False):
    if weights:
        remove_files(fedCNN.FEDERATED_LOCAL_WEIGHTS_PATH)

    if fed_model:
        remove_files(cNN.MODELS)

    if central_model:
        remove_files(cNN.CENTRALIZED_MODEL_PATH)


if __name__ == '__main__':
    main(weights=False, fed_model=True, central_model=False)
