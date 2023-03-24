from permuto_sdf_py.callbacks.callback import *
import os
import torch


class StateCallback(Callback):

    def __init__(self):
        pass

    def after_forward_pass(self, phase, loss, **kwargs):
        phase.iter_nr+=1
        phase.samples_processed_this_epoch+=1
        phase.loss_acum_per_epoch+=loss


    def epoch_started(self, phase, **kwargs):
        phase.loss_acum_per_epoch=0.0

    def epoch_ended(self, phase, model, save_checkpoint, checkpoint_path, **kwargs):

        phase.epoch_nr+=1

    def phase_started(self, phase, **kwargs):
        phase.samples_processed_this_epoch=0

    def phase_ended(self, phase, **kwargs):
        # phase.loader.reset()

        if(type(phase.loader) == torch.utils.data.DataLoader): # pytorchs dataloder has no reset 
            pass
        else:
            phase.loader.reset()
