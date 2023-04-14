from permuto_sdf_py.callbacks.callback import *
from torch.utils.tensorboard import SummaryWriter

class TensorboardCallback(Callback):

    def __init__(self, experiment_name):
        self.tensorboard_writer = SummaryWriter("tensorboard_logs/"+experiment_name)
        self.experiment_name=experiment_name
        

    def after_forward_pass(self, phase, loss=0, loss_rgb=0, loss_sdf_surface_area=0, loss_sdf_grad=0, loss_eikonal=0, loss_curvature=0,loss_lipshitz=0, lr=0, **kwargs):

        if phase.iter_nr%500==0:

            self.tensorboard_writer.add_scalar('permuto_sdf/' + phase.name + '/loss', loss.item(), phase.iter_nr)
            if loss_rgb!=0:
                self.tensorboard_writer.add_scalar('permuto_sdf/' + phase.name + '/loss_rgb', loss_rgb.item(), phase.iter_nr)

            if loss_sdf_grad!=0:
                self.tensorboard_writer.add_scalar('permuto_sdf/' + phase.name + '/loss_sdf_grad', loss_sdf_grad.item(), phase.iter_nr)
            if loss_eikonal!=0:
                self.tensorboard_writer.add_scalar('permuto_sdf/' + phase.name + '/loss_eik', loss_eikonal.item(), phase.iter_nr)
            if loss_curvature!=0:
                self.tensorboard_writer.add_scalar('permuto_sdf/' + phase.name + '/loss_curvature', loss_curvature.item(), phase.iter_nr)
            if loss_lipshitz!=0:
                self.tensorboard_writer.add_scalar('permuto_sdf/' + phase.name + '/loss_lipshitz', loss_lipshitz.item(), phase.iter_nr)
            


    def epoch_ended(self, phase, **kwargs):
        pass
        