from instant_ngp_2_py.callbacks.callback import *
import wandb
import hjson

class WandBCallback(Callback):

    def __init__(self, experiment_name, config_path, entity):
        self.experiment_name=experiment_name
        # loading the config file like this and giving it to wandb stores them on the website
        with open(config_path, 'r') as j:
            cfg = hjson.loads(j.read())
        # Before this init can be run, you have to use wandb login in the console you are starting the script from (https://docs.wandb.ai/ref/cli/wandb-login, https://docs.wandb.ai/ref/python/init)
        # entity= your username
        wandb.init(project=experiment_name, entity=entity,config = cfg)
        

    def after_forward_pass(self, phase, loss, loss_rgb, lr, loss_eikonal,**kwargs):

        # / act as seperators. If you would like to log train and test separately you would log test loss in test/loss 
        wandb.log({'train/loss': loss}, step=phase.iter_nr)
        wandb.log({'train/loss_eikonal': loss_eikonal}, step=phase.iter_nr)
        wandb.log({'train/lr': lr}, step=phase.iter_nr)
        if loss_rgb!=0:
            wandb.log({'train/loss_rgb': loss_rgb}, step=phase.iter_nr)


    def epoch_ended(self, phase, **kwargs):
        pass
        # mean_iou=phase.scores.avg_class_iou(print_per_class_iou=False)
        # self.tensorboard_writer.add_scalar('instant_ngp_2/' + phase.name + '/mean_iou', mean_iou, phase.epoch_nr)
        # self.vis.log(phase.epoch_nr, mean_iou, "iou_"+phase.name,  "loss_"+phase.name+"_"+self.experiment_name, smooth=False,  show_every=1)