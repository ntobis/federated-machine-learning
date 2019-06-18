import Centralized_CNN

def split_data_into_clients(num_of_clients, train_data, train_labels, test_data, test_labels):
    train_data = np.array_split(train_data, num_of_clients)
    train_labels = np.array_split(train_labels, num_of_clients)
    test_data = np.array_split(test_data, num_of_clients)
    test_labels = np.array_split(test_labels, num_of_clients)

    return train_data, train_labels, test_data, test_labels