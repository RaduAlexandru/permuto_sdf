import torchnet
import numpy as np
import torch

node_name="lnn"
port=8097
# logger_iou = torchnet.logger.VisdomPlotLogger('line', opts={'title': 'logger_iou'}, port=port, env='train_'+node_name)


class Vis():
    def __init__(self, env, port):
        self.port=port
        self.env=env
        # self.win_id="0"
        self.win_id=None

        self.name_dict=dict() #maps from name of the plot to the values that are stored currectly for that plot
        self.name2id_dict=dict() #maps from the name of the plot to the windows id
        self.logger_dict=dict()
        self.exp_alpha=0.03 #the lower the value the smoother the plot is

    def update_val(self, val, name, smooth):
        if name not in self.name_dict:
            self.name2id_dict[name] = str(len(self.name_dict))
            self.name_dict[name]=val
        else:
            if smooth:
                self.name_dict[name]= self.name_dict[name] + self.exp_alpha*(val-self.name_dict[name])
            else: 
                self.name_dict[name]=val
        
        return self.name_dict[name]

    def update_logger(self, x_axis, val, name_window, name_plot):
        if name_window not in self.logger_dict:
            self.logger_dict[name_window]=torchnet.logger.VisdomPlotLogger('line', opts={'title': name_window}, port=self.port, env=self.env, win=self.win_id)
            # self.logger_dict[name_window]=torchnet.logger.VisdomPlotLogger('line', opts={'title': name_window}, port=self.port, env=self.env, win=self.name2id_dict[name_plot] )
            print("started new line plot on win ", self.logger_dict[name_window].win)

        # print("update_logger val is ", val, "name plot is ", name_plot)
        self.logger_dict[name_window].log(x_axis, val, name=name_plot)

    def log(self, x_axis, val, name_window, name_plot, smooth, show_every=1, skip_first=0):
        if (x_axis<skip_first):
            return
        new_val=self.update_val(val,name_plot, smooth)
        if(x_axis%show_every==0):
            self.update_logger(x_axis, new_val, name_window, name_plot)

