import numpy as np
import tensorflow as tf

optimizers = tf.keras.optimizers  # like 'from tensorflow.keras import optimizers' (PyCharm import issue workaround)

# Per batch do:
# 1. Compute gradients for entire model
# 2. Get the gradients
# 3. Clip the gradients
# 4. Add noise to the gradients
# 5. Update the gradients in the model
# 6. Update the parameters in the model


class Sanitizer:
    def __init__(self):
        pass

    @staticmethod
    def sgd_clip_optimizer(lr=0.01, clipvalue=0.5):
        return optimizers.SGD(lr=lr, clipvalue=clipvalue)

    def Sanitize(self, gradients, noise_options):

        if noise_options.lower() is "normal":
            pass
        return gradients


if __name__ == '__main__':

    something = Sanitizer.Sanitize("NORMAL")
