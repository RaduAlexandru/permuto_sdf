from instant_ngp_2_py.callbacks.callback import *
from easypbr import Scene
import numpy as np

class ViewerCallback(Callback):

    def __init__(self):
        pass

    def after_forward_pass(self,**kwargs):
        pass
        # self.show_predicted_cloud(pred_softmax, cloud)
        # self.show_difference_cloud(pred_softmax, cloud)


    