from instant_ngp_2_py.callbacks.callback import *
from instant_ngp_2_py.callbacks.vis import *

class VisdomCallback(Callback):

    def __init__(self, experiment_name):
        self.vis=Vis("lnn", 8097)
        # self.iter_nr=0
        self.experiment_name=experiment_name

    def after_forward_pass(self, phase, loss, loss_dice, lr, pred_softmax, target, cloud, **kwargs):
        self.vis.log(phase.iter_nr, loss, "loss_"+phase.name,  "loss_"+phase.name+"_"+self.experiment_name, smooth=True,  show_every=10, skip_first=10)
        # self.vis.log(phase.iter_nr, loss_dice, "loss_dice_"+phase.name, "loss_"+phase.name+"_"+self.experiment_name, smooth=True,  show_every=10, skip_first=10)
        # if phase.grad:
            # self.vis.log(phase.iter_nr, lr, "lr", "loss_"+phase.name+"_"+self.experiment_name, smooth=False,  show_every=30)


    def epoch_ended(self, phase, **kwargs):
        mean_iou=phase.scores.avg_class_iou(print_per_class_iou=False)
        # self.vis.log(phase.epoch_nr, mean_iou, "iou_"+phase.name,  "loss_"+phase.name+"_"+self.experiment_name, smooth=False,  show_every=1)