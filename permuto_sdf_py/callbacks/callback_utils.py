from permuto_sdf  import TrainParams
from permuto_sdf_py.callbacks.callback import *
from permuto_sdf_py.callbacks.viewer_callback import *
from permuto_sdf_py.callbacks.visdom_callback import *
from permuto_sdf_py.callbacks.tensorboard_callback import *
from permuto_sdf_py.callbacks.wandb_callback import *
from permuto_sdf_py.callbacks.state_callback import *
from permuto_sdf_py.callbacks.phase import *



def create_callbacks(with_viewer, train_params, experiment_name, config_path):
    cb_list = []
    if(train_params.with_visdom()):
        cb_list.append(VisdomCallback(experiment_name))
    if(train_params.with_tensorboard()):
        tensorboard_callback=TensorboardCallback(experiment_name)
        cb_list.append(tensorboard_callback)
    if(train_params.with_wandb()):
        entity_name = "radualexandru" # your username on wandb
        cb_list.append(WandBCallback(experiment_name, config_path=config_path, entity=entity_name))
    if(with_viewer):
        cb_list.append(ViewerCallback())
    cb_list.append(StateCallback())
    cb = CallbacksGroup(cb_list)

    return cb

def create_callbacks_simple(with_viewer, with_tensorboard, experiment_name):
    cb_list = []
    if(with_tensorboard):
        tensorboard_callback=TensorboardCallback(experiment_name)
        cb_list.append(tensorboard_callback)
    if(with_viewer):
        cb_list.append(ViewerCallback())
    cb_list.append(StateCallback())
    cb = CallbacksGroup(cb_list)

    return cb
