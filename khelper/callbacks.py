import matplotlib.pyplot as plt
import pickle
from keras.callbacks import Callback

class PersistentHistory(Callback):
    """Same as the default Keras History object, but doesn't delete data when 
    new training begin. Also can load, save and plot itself."""

    def __init__(self):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def _plot_key(self, key):
        plt.plot(self.history[key])
        if 'val_'+key in self.history:
            plt.plot(self.history['val_'+key])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()

    def plot_loss(self):
        self._plot_key('loss')

    def plot_acc(self):
        self._plot_key('acc')

    def save(self, fp):
        try:
            with open(fp, 'wb') as f:
                pickle.dump(self.history,f)
        except:
            raise Exception('Saving history failed')

    @classmethod
    def load(cls, fp):
        history = cls()
        try:
            with open(fp, 'rb') as f:
                history.history = pickle.load(f)
            return history
        except:
            raise Exception('Loading history failed')

