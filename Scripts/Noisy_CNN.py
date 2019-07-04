import tensorflow as tf
from Scripts import Centralized_CNN as cNN


def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

# Training loop - using batches of 32
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
model = cNN.build_cnn()

global_step = tf.Variable(0)
for x, y in train_data:
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables),
                          global_step)
