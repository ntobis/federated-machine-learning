import tensorflow as tf
from tensorflow.python.keras import backend as K


def weighted_loss(y_true, y_pred):
    weights = 1 / 0.2
    return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, weights)


def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_function(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_function


class FocalLoss():
    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(self.alpha * K.pow(1. - pt_1, self.gamma) * K.log(pt_1)) - K.sum(
            (1 - self.alpha) * K.pow(pt_0, self.gamma) * K.log(1. - pt_0))


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.stopping_metric = []
        self.weights = []

    def __call__(self, weights, metric):
        if len(self.stopping_metric) == 0:
            pass
        elif min(self.stopping_metric) > metric:
            self.stopping_metric.clear()
            self.weights.clear()
        self.stopping_metric.append(metric)
        self.weights.append(weights)
        return True if len(self.stopping_metric) > self.patience else False

    def return_best_weights(self):
        print("Restoring best model weights.")
        return self.weights[0]


class AdditionalValidationSets(tf.keras.callbacks.Callback):
    def __init__(self, validation_sets, verbose=1, batch_size=None):
        """
        :param validation_sets:
        a list of 3-tuples (validation_data, validation_targets, validation_set_name)
        or 4-tuples (validation_data, validation_targets, sample_weights, validation_set_name)
        :param verbose:
        verbosity mode, 1 or 0
        :param batch_size:
        batch size to be used when evaluating on the additional datasets
        """
        super(AdditionalValidationSets, self).__init__()
        self.validation_sets = validation_sets
        for validation_set in self.validation_sets:
            if len(validation_set) not in [2, 3]:
                raise ValueError()
        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.batch_size = batch_size

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # evaluate on the additional validation sets
        for validation_set in self.validation_sets:
            if len(validation_set) == 3:
                validation_data, validation_targets, validation_set_name = validation_set
                sample_weights = None
            elif len(validation_set) == 4:
                validation_data, validation_targets, sample_weights, validation_set_name = validation_set
            else:
                raise ValueError()

            results = self.model.evaluate(x=validation_data,
                                          y=validation_targets,
                                          verbose=self.verbose,
                                          sample_weight=sample_weights,
                                          batch_size=self.batch_size)

            for i, result in enumerate(results):
                if i == 0:
                    valuename = validation_set_name + '_loss'
                else:
                    valuename = validation_set_name + '_' + self.model.metrics[i - 1].name
                self.history.setdefault(valuename, []).append(result)
