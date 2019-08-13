import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np


# Generate Data
train_data = np.random.randint(0, 100, size=(10, 215, 215, 1))
train_labels = np.random.randint(0, 2, size=(10, 1))


# Set up metrics
metrics = ['accuracy',
           tf.keras.metrics.TruePositives(),
           tf.keras.metrics.TrueNegatives(),
           tf.keras.metrics.FalseNegatives(),
           tf.keras.metrics.FalsePositives()]

# Set up model type
model = models.Sequential(name='CNN')

# Add layers
model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), input_shape=train_data[0].shape))
model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2)))
model.add(layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(units=128))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.Dense(units=1, activation='sigmoid'))
model.compile('sgd', 'binary_crossentropy', metrics)

history = model.fit(train_data, train_labels, batch_size=1, epochs=30)