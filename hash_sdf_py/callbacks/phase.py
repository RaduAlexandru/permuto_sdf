#https://github.com/devforfu/pytorch_playground/blob/master/loop.ipynbA

from instant_ngp_2_py.callbacks.scores import *

class Phase:
    """
    Model training loop phase.

    Each model's training loop iteration could be separated into (at least) two
    phases: training and validation. The instances of this class track
    metrics and counters, related to the specific phase, and keep the reference
    to subset of data, used during phase.
    """

    def __init__(self, name, loader, grad):
        self.name = name
        self.loader = loader
        self.grad = grad
        self.iter_nr = 0
        self.epoch_nr = 0
        self.samples_processed_this_epoch = 0
        self.scores= Scores()
        self.loss_acum_per_epoch=0.0
        # self.batch_loss = None
        # self.batch_index = 0
        # self.rolling_loss = 0
        # self.losses = []
        # self.metrics = OrderedDict()

    # @property
    # def last_loss(self):
    #     return self.losses[-1] if self.losses else None

    # @property
    # def last_metrics(self):
    #     metrics = OrderedDict()
    #     metrics[f'{self.name}_loss'] = self.last_loss
    #     for name, values in self.metrics.items():
    #         metrics[f'{self.name}_{name}'] = values[-1]
    #     return metrics

    # @property
    # def metrics_history(self):
    #     metrics = OrderedDict()
    #     for name, values in self.metrics.items():
    #         metrics[f'{self.name}_{name}'] = values
    #     return metrics

    # def update(self, loss):
        # self.losses.append(loss)

    # def update_metric(self, name, value):
    #     if name not in self.metrics:
    #         self.metrics[name] = []
    #     self.metrics[name].append(value)