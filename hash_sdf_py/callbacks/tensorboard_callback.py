from instant_ngp_2_py.callbacks.callback import *
from torch.utils.tensorboard import SummaryWriter

class TensorboardCallback(Callback):

    def __init__(self, experiment_name):
        self.tensorboard_writer = SummaryWriter("tensorboard_logs/"+experiment_name)
        self.experiment_name=experiment_name
        

    def after_forward_pass(self, phase, loss=0, loss_rgb=0, loss_sdf_surface_area=0, loss_sdf_grad=0, loss_eikonal=0, loss_curvature=0,loss_lipshitz=0,  neus_variance_mean=0, lr=0, **kwargs):
        # self.vis.log(phase.iter_nr, loss, "loss_"+phase.name,  "loss_"+phase.name+"_"+self.experiment_name, smooth=True,  show_every=10, skip_first=10)
        # self.vis.log(phase.iter_nr, loss_dice, "loss_dice_"+phase.name, "loss_"+phase.name+"_"+self.experiment_name, smooth=True,  show_every=10, skip_first=10)
        # if phase.grad:
            # self.vis.log(phase.iter_nr, lr, "lr", "loss_"+phase.name+"_"+self.experiment_name, smooth=False,  show_every=30)

        self.tensorboard_writer.add_scalar('instant_ngp_2/' + phase.name + '/lr', lr, phase.iter_nr)

        if phase.iter_nr%100==0:

            self.tensorboard_writer.add_scalar('instant_ngp_2/' + phase.name + '/loss', loss, phase.iter_nr)
            if loss_rgb!=0:
                self.tensorboard_writer.add_scalar('instant_ngp_2/' + phase.name + '/loss_rgb', loss_rgb, phase.iter_nr)

            # if loss_sdf_surface_area!=0:
                # self.tensorboard_writer.add_scalar('instant_ngp_2/' + phase.name + '/loss_sdf_surface_area', loss_sdf_surface_area, phase.iter_nr)
            if loss_sdf_grad!=0:
                self.tensorboard_writer.add_scalar('instant_ngp_2/' + phase.name + '/loss_sdf_grad', loss_sdf_grad, phase.iter_nr)
            if loss_eikonal!=0:
                self.tensorboard_writer.add_scalar('instant_ngp_2/' + phase.name + '/loss_eik', loss_eikonal, phase.iter_nr)
            if loss_curvature!=0:
                self.tensorboard_writer.add_scalar('instant_ngp_2/' + phase.name + '/loss_curvature', loss_curvature, phase.iter_nr)
            if loss_lipshitz!=0:
                self.tensorboard_writer.add_scalar('instant_ngp_2/' + phase.name + '/loss_lipshitz', loss_lipshitz, phase.iter_nr)
            if neus_variance_mean!=0 and neus_variance_mean is not None:
                self.tensorboard_writer.add_scalar('instant_ngp_2/' + phase.name + '/neus_variance_mean', neus_variance_mean, phase.iter_nr)
            


    def epoch_ended(self, phase, **kwargs):
        pass
        # mean_iou=phase.scores.avg_class_iou(print_per_class_iou=False)
        # self.tensorboard_writer.add_scalar('instant_ngp_2/' + phase.name + '/mean_iou', mean_iou, phase.epoch_nr)
        # self.vis.log(phase.epoch_nr, mean_iou, "iou_"+phase.name,  "loss_"+phase.name+"_"+self.experiment_name, smooth=False,  show_every=1)