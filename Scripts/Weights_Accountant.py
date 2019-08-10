import numpy as np


class WeightsAccountant:
    def __init__(self, weights):
        self.Global_Weights = weights
        self.Local_Weights = []
        self.Localized_Layers = {}

    def append_localized_layer(self, client, weights):
        self.Localized_Layers[client] = weights

    def append_local_weights(self, weights):
        self.Local_Weights.append(weights)

    def average_local_weights(self):
        self.Local_Weights = np.array(self.Local_Weights).T
        self.Global_Weights = np.mean(self.Local_Weights, axis=1)
        del self.Local_Weights
        self.Local_Weights = []
        return self.Global_Weights

    def get_global_weights(self):
        return self.Global_Weights

    def get_localized_layers(self, client):
        return self.Localized_Layers[client]

    def is_localized(self, client):
        return client in self.Localized_Layers

    def get_client_weights(self, client=None):
        client = client if client is not None else next(iter(self.Localized_Layers))
        return np.concatenate((self.Global_Weights, self.Localized_Layers[client]))
