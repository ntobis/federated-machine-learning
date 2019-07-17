import numpy as np
import tensorflow as tf


class WeightsAccountant:
    def __init__(self, weights, clients):
        self.Global_Weights = weights
        self.Local_Weights = []
        self.Local_Updates = []
        self.Local_Norms = []
        self.Clipped_Updates = None
        self.Global_Updates = None
        self.median = None
        self.num_clients = clients

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

    def compute_local_updates(self):
        for local_model in self.Local_Weights:
            updates = [loc - glob for glob, loc in zip(self.Global_Weights, local_model)]
            self.Local_Updates.append(updates)

        # layer_1 = 0
        # layer_2 = 0
        # layer_3 = 0
        # layer_4 = 0
        # layer_5 = 0
        # layer_6 = 0
        # layer_7 = 0
        # layer_8 = 0
        #
        # for i, update in enumerate(self.Local_Updates):
        #     layer_1 += update[0]
        #     layer_2 += update[1]
        #     layer_3 += update[2]
        #     layer_4 += update[3]
        #     layer_5 += update[4]
        #     layer_6 += update[5]
        #     layer_7 += update[6]
        #     layer_8 += update[7]
        #
        # print(np.sum(layer_1))
        # print(np.sum(layer_2))
        # print(np.sum(layer_3))
        # print(np.sum(layer_4))
        # print(np.sum(layer_5))
        # print(np.sum(layer_6))
        # print(np.sum(layer_7))
        # print(np.sum(layer_8))

    def compute_update_norm(self):
        self.Local_Norms = []
        for update in self.Local_Updates:
            l2_norm = [np.sqrt(np.sum(np.square(layer))) for layer in update]
            self.Local_Norms.append(l2_norm)
        self.Local_Norms = np.array(self.Local_Norms)

    def clip_updates(self):
        self.median = np.median(self.Local_Norms, axis=0)
        scaling_factor = np.divide(self.Local_Norms, self.median, where=self.median > 0)
        scaling_factor = np.maximum(scaling_factor, 1)
        self.Clipped_Updates = np.divide(self.Local_Updates, scaling_factor)

    def reset_local_parameters(self):
        del self.Local_Weights
        del self.Local_Updates
        del self.Local_Norms
        del self.Clipped_Updates
        self.Local_Weights = []
        self.Local_Updates = []
        self.Local_Norms = []
        self.Clipped_Updates = []

    def average_local_clipped_updates(self):
        self.Local_Updates = np.array(self.Clipped_Updates).T
        self.Global_Updates = np.mean(self.Local_Updates, axis=1)
        self.reset_local_parameters()

    def compute_noisy_global_weights(self, sigma):

        for i in range(len(self.Global_Weights)):
            noise = np.random.normal(loc=0.0, scale=sigma * self.median[i], size=self.Global_Updates[i].shape) / self.num_clients
            self.Global_Weights[i] = self.Global_Weights[i] + self.Global_Updates[i] + noise

