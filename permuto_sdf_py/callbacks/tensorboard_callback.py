from instant_ngp_2_py.callbacks.callback import *
from torch.utils.tensorboard import SummaryWriter

class TensorboardCallback(Callback):

    def __init__(self, experiment_name):
        self.tensorboard_writer = SummaryWriter("tensorboard_logs/"+experiment_name)
        self.experiment_name=experiment_name
        

    def after_forward_pass(self, phase, loss=0, loss_rgb=0, loss_sdf_surface_area=0, loss_sdf_grad=0, loss_eikonal=0, loss_curvature=0,loss_lipshitz=0,  neus_variance_mean=0, lr=0, **kwargs):

        if phase.iter_nr%100==0:

            self.tensorboard_writer.add_scalar('permuto_sdf/' + phase.name + '/loss', loss, phase.iter_nr)
            if loss_rgb!=0:
                self.tensorboard_writer.add_scalar('permuto_sdf/' + phase.name + '/loss_rgb', loss_rgb, phase.iter_nr)

            if loss_sdf_grad!=0:
                self.tensorboard_writer.add_scalar('permuto_sdf/' + phase.name + '/loss_sdf_grad', loss_sdf_grad, phase.iter_nr)
            if loss_eikonal!=0:
                self.tensorboard_writer.add_scalar('permuto_sdf/' + phase.name + '/loss_eik', loss_eikonal, phase.iter_nr)
            if loss_curvature!=0:
                self.tensorboard_writer.add_scalar('permuto_sdf/' + phase.name + '/loss_curvature', loss_curvature, phase.iter_nr)
            if loss_lipshitz!=0:
                self.tensorboard_writer.add_scalar('permuto_sdf/' + phase.name + '/loss_lipshitz', loss_lipshitz, phase.iter_nr)
            if neus_variance_mean!=0 and neus_variance_mean is not None:
                self.tensorboard_writer.add_scalar('permuto_sdf/' + phase.name + '/neus_variance_mean', neus_variance_mean, phase.iter_nr)
            


    def epoch_ended(self, phase, **kwargs):
        pass
        