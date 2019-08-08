# Imports
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tensorflow as tf
import numpy as np

from Scripts._old import Centralized_CNN as cNN
from Scripts import Data_Loader_Functions as Data_Loader
from Scripts import Print_Functions as Output


def shuffling(train_data, train_labels):
    shuffled_inputs = train_data.copy()
    shuffled_outputs = train_labels.copy()
    rng_state = np.random.get_state()
    np.random.shuffle(shuffled_inputs)
    np.random.set_state(rng_state)
    np.random.shuffle(shuffled_outputs)

    return shuffled_inputs, shuffled_outputs


def loss(model, x, y_true):
    y_pred = model(x)
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def train(model, train_data, train_labels, batch_size, epochs, optimizer, shuffle=True, noise=None, clipnorm=None):
    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    for epoch in range(epochs):
        epoch_loss_avg = tf.metrics.Mean()
        epoch_accuracy = tf.metrics.Accuracy()

        # Shuffle
        if shuffle:
            train_data, train_labels = shuffling(train_data, train_labels)
        input_batches = np.array_split(train_data, train_data.shape[0] // batch_size)
        output_batches = np.array_split(train_labels, train_labels.shape[0] // batch_size)

        global_step = tf.Variable(0)
        for input_batch, output_batch in zip(input_batches, output_batches):
            # Optimize the model
            loss_value, grads = grad(model, input_batch, output_batch)

            for idx, gradient in enumerate(grads):

                # Clip gradients
                if clipnorm:
                    grads[idx] = tf.clip_by_norm(grads[idx], clipnorm)

                # Add noise
                if noise['type'].lower() is 'normal' or 'gaussian':
                    grads[idx] += tf.random.normal(gradient.shape, noise['mean'], noise['stdev'] * clipnorm)
                elif noise['type'].lower() is 'laplace':
                    grads[idx] += np.random.laplace(noise['mean'], noise['stdev'] * clipnorm, gradient.shape)
            optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)

            # Track progress
            epoch_loss_avg(loss_value)  # add current batch loss
            # compare predicted label to actual label
            epoch_accuracy(tf.argmax(model(input_batch), axis=1, output_type=tf.int32), output_batch)

            # end epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 1 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch + 1,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))

    return model


def main():
    noisy_model = cNN.build_cnn(input_shape=(28, 28, 1))
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    train_data, train_labels, test_data, test_labels, dataset = Data_Loader.load_data("MNIST")

    test_data = tf.convert_to_tensor(test_data, dtype=tf.float32)
    test_labels = tf.convert_to_tensor(test_labels, dtype=tf.float32)

    noise = {'type': 'normal', 'mean': 0, 'stdev': 4}
    noisy_model = train(model=noisy_model,
                        train_data=train_data,
                        train_labels=train_labels,
                        batch_size=32,
                        epochs=10,
                        optimizer=optimizer,
                        noise=noise,
                        clipnorm=3)

    noisy_model.compile(optimizer=optimizer,
                        loss=tf.keras.losses.sparse_categorical_crossentropy,
                        metrics=['accuracy'])

    test_loss, test_acc = cNN.evaluate_cnn(noisy_model, test_data, test_labels)
    Output.print_loss_accuracy(test_acc, test_loss)


if __name__ == '__main__':
    main()
