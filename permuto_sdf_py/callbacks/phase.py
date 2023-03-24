#https://github.com/devforfu/pytorch_playground/blob/master/loop.ipynbA

# from permuto_sdf_py.callbacks.scores import *

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
        # self.scores= Scores()
        self.loss_acum_per_epoch=0.0
    