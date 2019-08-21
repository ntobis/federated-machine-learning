from collections import defaultdict

import numpy as np


class WeightsAccountant:
    def __init__(self, model):
        self.client_weights = {}
        self.default_weights = {}
        self.define_default_weights(model)
        self.shared_weights = defaultdict(list)

    def get_client_weights(self):
        return self.client_weights

    def set_client_weights(self, client_weights):
        self.client_weights = client_weights

    def apply_client_weights(self, model, client):
        print("Setting client {} weights:".format(client), end=" ")
        if client in self.client_weights:
            for layer in model.layers:
                if layer.name in self.client_weights[client]:
                    layer.set_weights(self.client_weights[client][layer.name])
                    print(layer.name, end=" ")
        else:
            self.apply_default_weights(model)
        print()

    def apply_default_weights(self, model):
        print("Setting default weights:", end=" ")
        for layer in model.layers:
            if layer.name in self.default_weights:
                layer.set_weights(self.default_weights[layer.name])
                print(layer.name, end=" ")
        print()

    def update_client_weights(self, model, client):
        print("Updating client {}:".format(client), end=" ")
        weights = {}
        for layer in model.layers:
            if len(layer.get_weights()) > 0:
                weights[layer.name] = layer.get_weights()
                print(layer.name, end=" ")
        self.client_weights[client] = weights
        print()

    def define_default_weights(self, model):
        print("Define default weights:", end=" ")
        for layer in model.layers:
            if len(layer.get_weights()) > 0:
                self.default_weights[layer.name] = layer.get_weights()
                print(layer.name, end=" ")
        print()

    def update_default_weights(self):
        print("Update default weights:", end=" ")
        for layer_name, weights in self.shared_weights.items():
            if layer_name in self.default_weights:
                self.default_weights[layer_name] = weights
                print(layer_name, end=" ")
        print()

    # Federated averaging functions
    def determine_shared_weights(self, layer_type):
        layer_type = '' if layer_type is None else layer_type
        print("Determining shared weights:")
        for client, client_layers in self.client_weights.items():
            print("Client {}: ".format(client), end=" ")
            for layer_name, weights in client_layers.items():
                if layer_type in layer_name:
                    self.shared_weights[layer_name].append(weights)
                    print(layer_name, end=" ")
            print()

    def average_shared_weights(self):
        print("Averaging shared weights:", end=" ")
        for layer_name, weights in self.shared_weights.items():
            # noinspection PyTypeChecker
            self.shared_weights[layer_name] = np.mean(weights, axis=0)
            print(layer_name, end=" ")
        print()

    def distribute_shared_weights_to_clients(self):
        print("Distribute shared weights:")
        for client, client_layers in self.client_weights.items():
            print("Client {}:".format(client), end=" ")
            for shared_layer_name, weights in self.shared_weights.items():
                client_layers[shared_layer_name] = weights
                print(shared_layer_name, end=" ")
            print()

    def print_client_update(self):
        print("Clients with localized weights:", end=" ")
        print(sorted(self.client_weights.keys()))
        equal = [all([np.array_equal(client_layers[layer_name], self.default_weights[layer_name])
                      for layer_name in client_layers.keys()])
                 for client_layers in self.client_weights.values()]
        print("All equal: {}".format(all(equal)))

    def federated_averaging(self, layer_type=None):
        self.determine_shared_weights(layer_type)
        self.average_shared_weights()
        self.distribute_shared_weights_to_clients()

        if layer_type is None:
            self.update_default_weights()

        del self.shared_weights
        self.shared_weights = defaultdict(list)
