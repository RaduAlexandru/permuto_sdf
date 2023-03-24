from permuto_sdf_py.callbacks.callback import *
from permuto_sdf_py.callbacks.vis import *

class VisdomCallback(Callback):

    def __init__(self, experiment_name):
        self.vis=Vis("lnn", 8097)
        self.experiment_name=experiment_name

    def after_forward_pass(self, phase, loss, loss_dice, lr, pred_softmax, target, cloud, **kwargs):
        self.vis.log(phase.iter_nr, loss, "loss_"+phase.name,  "loss_"+phase.name+"_"+self.experiment_name, smooth=True,  show_every=10, skip_first=10)
        


    def epoch_ended(self, phase, **kwargs):
        mean_iou=phase.scores.avg_class_iou(print_per_class_iou=False)