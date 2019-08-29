import math
import os
import sys

# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------ Paths ----------------------------------------------------- #

ROOT = os.path.dirname(os.path.dirname(__file__))
FIGURES = os.path.join(ROOT, "Figures")
RESULTS = os.path.join(ROOT, "Results")


# ---------------------------------------------------- End Paths --------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------- Print Functions ------------------------------------------------ #

def print_communication_round(com_round):
    print()
    print("-" * 131)
    print("{} Communication Round {} {}".format("-" * math.floor((130 - 21 - len(str(com_round))) / 2), com_round,
                                                "-" * math.ceil((130 - 21 - len(str(com_round))) / 2)))


def print_client_id(client_id):
    print()
    print("{} Client {} {}".format("-" * math.floor((130 - 8 - len(str(client_id))) / 2), client_id,
                                   "-" * math.ceil((130 - 8 - len(str(client_id))) / 2)))


def print_loss_accuracy(accuracy, loss, data_type="Test"):
    print("-----------------------")
    print("{} Loss: {:5.2f}".format(data_type, loss))
    print("{} Accuracy: {:5.2f}%".format(data_type, 100 * accuracy))
    print("-----------------------")
    print()


def print_session(sess):
    print("\n\n\033[1m{} Session {} {}\033[0m".format("-" * math.floor((130 - 9 - len(str(sess))) / 2),
                                                      sess,
                                                      "-" * math.ceil((130 - 9 - len(str(sess))) / 2)))


def print_shard(percentage):
    print("\n\n\033[1m{} Shard {:.0%} {}\033[0m".format("-" * math.floor((130 - 7 - len(str(percentage))) / 2),
                                                        percentage,
                                                        "-" * math.ceil((130 - 7 - len(str(percentage))) / 2)))


def print_experiment(experiment):
    print("\n\n\033[1m{} Experiment {} {}\033[0m".format("-" * math.floor((130 - 12 - len(experiment)) / 2), experiment,
                                                         "-" * math.ceil((130 - 12 - len(experiment)) / 2)))


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# ----------------------------------------------- End Print Functions ---------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #
