from keras import callbacks


class Evaluate(callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.history = None

    def on_epoch_end(self, epoch, logs=None):
        pass
