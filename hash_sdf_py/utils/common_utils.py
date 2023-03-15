import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
import functools
from matplotlib import cm
import random
import os

import copy
import inspect
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch as th
import torch.nn.functional as thf
from torch import Tensor
from torch.nn.utils.weight_norm import WeightNorm, remove_weight_norm
from torch.nn.modules.utils import _pair

from functools import reduce
from torch.nn.modules.module import _addindent

from easypbr  import *

from hash_sdf_py.utils.aabb import *
from hash_sdf  import Sphere

from hash_sdf_py.paths.data_paths import *


#Just to have something close to the macros we have in c++
def profiler_start(name):
    if(Profiler.is_profiling_gpu()):
        torch.cuda.synchronize()
    Profiler.start(name)
def profiler_end(name):
    if(Profiler.is_profiling_gpu()):
        torch.cuda.synchronize()
    Profiler.end(name)
TIME_START = lambda name: profiler_start(name)
TIME_END = lambda name: profiler_end(name)







#from nerfies
def cosine_easing_window(num_freqs, alpha):
    """Eases in each frequency one by one with a cosine.
    This is equivalent to taking a Tukey window and sliding it to the right
    along the frequency spectrum.
    Args:
    num_freqs: the number of frequencies.
    alpha: will ease in each frequency as alpha goes from 0.0 to num_freqs.
    Returns:
    A 1-d numpy array with num_sample elements containing the window.
    """
    x = torch.clip(alpha - torch.arange(num_freqs, dtype=torch.float32), 0.0, 1.0)
    return 0.5 * (1 + torch.cos(math.pi * x + math.pi))




eps = 1e-8

#wraps module, and changes them to become a torchscrip version of them during inference
class TorchScriptTraceWrapper(torch.nn.Module):
    def __init__(self, module):
        super(TorchScriptTraceWrapper, self).__init__()

        self.module=module
        self.module_traced=None

    def forward(self, *args):
        args_list=[]
        for arg in args:
            args_list.append(arg)
        if self.module_traced==None:
                self.module_traced = torch.jit.trace(self.module, args_list )
        return self.module_traced(*args)




def nchw2nhwc(x):
    x=x.permute(0,2,3,1)
    return x

def nhwc2nchw(x):
    x=x.permute(0,3,1,2)
    return x

#make from N,C,H,W to N,Nrpixels,C
def nchw2nXc(x):
    nr_feat=x.shape[1]
    nr_batches=x.shape[0]
    x=x.permute(0,2,3,1) #from N,C,H,W to N,H,W,C
    x=x.view(nr_batches, -1, nr_feat)
    return x

#make from N,NrPixels,C to N,C,H,W
def nXc2nchw(x, h, w):
    nr_feat=x.shape[2]
    nr_batches=x.shape[0]
    x=x.view(nr_batches, h, w, nr_feat)
    x=x.permute(0,3,1,2) #from N,H,W,C, N,C,H,W
    return x

# make from N,C,H,W to Nrpixels,C ONLY works when N is 1
def nchw2lin(x):
    # if x.shape[0]!=1:
        # print("nchw2lin supposes that the N is 1 however x has shape ", x.shape )
        # exit(1)
    nr_feat=x.shape[1]
    nr_batches=x.shape[0]
    x=x.permute(0,2,3,1).contiguous() #from N,C,H,W to N,H,W,C
    x=x.view(-1, nr_feat)
    return x

#go from nr_pixels, C to 1,C,H,W
def lin2nchw(x, h, w):
    nr_feat=x.shape[1]
    x=x.view(1, h, w, nr_feat)
    x=nhwc2nchw(x)
    return x

def img2tex(x):
    x=x.permute(0,2,3,1).squeeze(0) #nchw to hwc
    return x

def tex2img(x):
    x=x.unsqueeze(0).permute(0,3,1,2)  #nhwc to  nchw
    return x

#https://github.com/NVlabs/instant-ngp/blob/c3f3534801704fe44585e6fbc2dc5f528e974b6e/scripts/common.py#L139
def srgb_to_linear(img):
	limit = 0.04045
	return torch.where(img > limit, torch.pow((img + 0.055) / 1.055, 2.4), img / 12.92)

def linear_to_srgb(img):
	limit = 0.0031308
	return torch.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)




def map_range_tensor( input_val, input_start, input_end,  output_start,  output_end):
    # input_clamped=torch.clamp(input_val, input_start, input_end)
    # input_clamped=max(input_start, min(input_end, input_val))
    input_clamped=torch.clamp(input_val, input_start, input_end)
    return output_start + ((output_end - output_start) / (input_end - input_start)) * (input_clamped - input_start)

def map_range_val( input_val, input_start, input_end,  output_start,  output_end):
    # input_clamped=torch.clamp(input_val, input_start, input_end)
    input_clamped=max(input_start, min(input_end, input_val))
    # input_clamped=torch.clamp(input_val, input_start, input_end)
    return output_start + ((output_end - output_start) / (input_end - input_start)) * (input_clamped - input_start)

def map_range_np( input_val, input_start, input_end,  output_start,  output_end):
    # input_clamped=torch.clamp(input_val, input_start, input_end)
    input_clamped=np.clip(input_val, input_start, input_end)
    # input_clamped=torch.clamp(input_val, input_start, input_end)
    return output_start + ((output_end - output_start) / (input_end - input_start)) * (input_clamped - input_start)



def smoothstep_tensor(e0, e1, x):
    t = torch.clip(((x - e0) / (e1 - e0)), 0, 1)
    return t * t * (3.0 - 2.0 * t)


def smootherstep_tensor(e0, e1, x):
    t = torch.clip(((x - e0) / (e1 - e0)), 0, 1)
    return (t ** 3) * (t * (t * 6 - 15) + 10)

def smoothstep_val(e0, e1, x):
    t = np.clip(((x - e0) / (e1 - e0)), 0, 1)
    return t * t * (3.0 - 2.0 * t)


def smootherstep_val(e0, e1, x):
    t = np.clip(((x - e0) / (e1 - e0)), 0, 1)
    return (t ** 3) * (t * (t * 6 - 15) + 10)

#get a parameter t from 0 to 1 and maps it to range 0, 1 but with a very fast increase at the begining and it stops slowly towards 1. From Fast and Funky 1D Nonlinear Transformations
def smoothstop2(t):
    return 1-pow((1-t),2)
def smoothstop3(t):
    return 1-pow((1-t),3)
def smoothstop4(t):
    return 1-pow((1-t),4)
def smoothstop5(t):
    return 1-pow((1-t),5)
def smoothstop_n(t,n):
    return 1-pow((1-t),n)


#make it  a power of 2
def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

# Compute a power of two less than or equal to `n`
def previous_power_of_2(n):

    # set all bits after the last set bit
    n = n | (n >> 1)
    n = n | (n >> 2)
    n = n | (n >> 4)
    n = n | (n >> 8)
    n = n | (n >> 16)

    # drop all but the last set bit from `n`
    return n - (n >> 1)


def show_points(points, name, translation=[0,0,0], color=None, color_per_vert=None, normal_per_vert=None):
    pred_points_cpu=points.contiguous().view(-1,3).detach().double().cpu().numpy()
    pred_strands_mesh=Mesh()
    pred_strands_mesh.V=pred_points_cpu
    pred_strands_mesh.m_vis.m_show_points=True
    pred_strands_mesh.model_matrix.translate(translation)
    if color!=None:
        pred_strands_mesh.m_vis.m_point_color=color
    if color_per_vert!=None:
        pred_strands_mesh.C=color_per_vert.contiguous().view(-1,3).detach().double().cpu().numpy()
        pred_strands_mesh.m_vis.set_color_pervertcolor()
    if normal_per_vert!=None:
        pred_strands_mesh.NV=normal_per_vert.contiguous().view(-1,3).detach().double().cpu().numpy()
    Scene.show(pred_strands_mesh, name)
    return pred_strands_mesh


def colormap(values, colormap_name):
    values_np=values.detach().cpu().numpy()
    colormap = cm.get_cmap(colormap_name)
    values_np_colored = colormap(values_np)
    # values_np_colored=cm.colors.to_rgb(values_np_colored) #drop the alpha channel
    colors=torch.from_numpy(values_np_colored)
    colors=colors.squeeze()[:,0:3]

    return colors
    


def leaky_relu_init(m, negative_slope=0.2):

    gain = np.sqrt(2.0 / (1.0 + negative_slope ** 2))

    if isinstance(m, th.nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, th.nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, th.nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // 2
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, th.nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // 4
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, th.nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] // 8
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, th.nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * np.sqrt(2.0 / (n1 + n2))
    else:
        return

  

    m.weight.data.uniform_(-std * np.sqrt(3.0), std * np.sqrt(3.0))
    if m.bias is not None:
        m.bias.data.zero_()

    if isinstance(m, th.nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    

    # m.weights_initialized=True

def apply_weight_init_fn(m, fn, negative_slope=1.0):

    should_initialize_weight=True
    if not hasattr(m, "weights_initialized"): #if we don't have this then we need to intiialzie
        # fn(m, is_linear, scale)
        should_initialize_weight=True
    elif m.weights_initialized==False: #if we have it but it's set to false
        # fn(m, is_linear, scale)
        should_initialize_weight=True
    else:
        print("skipping weight init on ", m)
        should_initialize_weight=False

    if should_initialize_weight:
        # fn(m, is_linear, scale)
        fn(m,negative_slope)
        # m.weights_initialized=True
        for module in m.children():
            apply_weight_init_fn(module, fn, negative_slope)






# https://github.com/pytorch/pytorch/issues/34704#issuecomment-1000310792
#grid sample 3d that support double backward
def grid_sample_3d(image, optical):
    N, C, ID, IH, IW = image.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1);
    iy = ((iy + 1) / 2) * (IH - 1);
    iz = ((iz + 1) / 2) * (ID - 1);
    with torch.no_grad():
        
        ix_tnw = torch.floor(ix);
        iy_tnw = torch.floor(iy);
        iz_tnw = torch.floor(iz);

        ix_tne = ix_tnw + 1;
        iy_tne = iy_tnw;
        iz_tne = iz_tnw;

        ix_tsw = ix_tnw;
        iy_tsw = iy_tnw + 1;
        iz_tsw = iz_tnw;

        ix_tse = ix_tnw + 1;
        iy_tse = iy_tnw + 1;
        iz_tse = iz_tnw;

        ix_bnw = ix_tnw;
        iy_bnw = iy_tnw;
        iz_bnw = iz_tnw + 1;

        ix_bne = ix_tnw + 1;
        iy_bne = iy_tnw;
        iz_bne = iz_tnw + 1;

        ix_bsw = ix_tnw;
        iy_bsw = iy_tnw + 1;
        iz_bsw = iz_tnw + 1;

        ix_bse = ix_tnw + 1;
        iy_bse = iy_tnw + 1;
        iz_bse = iz_tnw + 1;

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);


    with torch.no_grad():

        torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
        torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
        torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

        torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
        torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
        torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

        torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
        torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
        torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

        torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
        torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
        torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

        torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

    image = image.view(N, C, ID * IH * IW)

    tnw_val = torch.gather(image, 2, (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tne_val = torch.gather(image, 2, (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tsw_val = torch.gather(image, 2, (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tse_val = torch.gather(image, 2, (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bnw_val = torch.gather(image, 2, (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(image, 2, (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(image, 2, (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(image, 2, (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    out_val = (tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
               tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
               tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
               tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
               bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
               bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
               bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
               bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W))

    return out_val


#from https://arxiv.org/pdf/2111.15135.pdf
def gauss_activ(x, sigma):
    gaus_val=torch.exp(  -(x*x)/ (2*sigma*sigma)   )

    # if normalize: #this si bugged I think since it doesn't actually normalize
        # normalization=sigma*np.sqrt(2*math.pi)
        # gaus_val=gaus_val/normalization
    return gaus_val


#summary
def summary(model,file=sys.stderr):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            # print("name is ", name)
            if p is not None:
                # print("parameter with shape ", p.shape)
                # print("parameter has dim ", p.dim())
                if p.dim()==0: #is just a scalar parameter
                    total_params+=1
                else:
                    total_params += reduce(lambda x, y: x * y, p.shape)
                # if(p.grad==None):
                #     print("p has no grad", name)
                # else:
                #     print("p has gradnorm ", name ,p.grad.norm() )

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stderr:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
            for name, p in model._parameters.items():
                if hasattr(p, 'grad'):
                    if(p.grad==None):
                        print("p has no grad", name)
                        main_str+="p no grad"
                    else:
                        # print("p has gradnorm ", name ,p.grad.norm() )
                        main_str+= "\n" + name + " p has grad norm, min, max" + str(p.grad.norm()) + " " + str(p.grad.min()) + " " + str(p.grad.max())
                        main_str+= "\n" + name + " p has grad type" + str(p.grad.type()) 

                        #check for nans
                        if torch.isnan(p.grad).any():
                            print("NAN detected in grad of ", name)
                            print("main_str is ", main_str)
                            exit(1)

                #show also the parameter itself, an not only the gradient
                if (p is not None):
                    if (p.numel()!=0):
                        main_str+= "\n" + name + " Param, min, max"  + str(p.min()) + " " + str(p.max())

                        #check for nans
                        if torch.isnan(p).any():
                            print("NAN detected in param ", name)
                            print("main_str is ", main_str)
                            exit(1)

        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        print(string, file=file)
    return count


def create_dataloader(config_path, dataset_name, scene_name, low_res, comp_name, with_mask):

    from dataloaders import DataLoaderEasyPBR
    from dataloaders import DataLoaderMultiFace
    from dataloaders import DataLoaderPhenorobCP1
    from dataloaders import DataLoaderDTU

    data_path=data_paths[comp_name]
    
    if dataset_name=="easypbr":
        # easypbr
        loader_train=DataLoaderEasyPBR(config_path)
        loader_train.set_mode_train()
        loader_train.set_limit_to_nr_imgs(-1)
        loader_train.set_load_mask(with_mask)
        loader_train.set_dataset_path(os.path.join(data_path,"easy_pbr_renders"))
        if low_res:
            loader_train.set_subsample_factor(4)
        loader_train.start()
        #easypbr test
        loader_test=DataLoaderEasyPBR(config_path)
        loader_test.set_mode_test()
        loader_test.set_limit_to_nr_imgs(10)
        loader_test.set_load_mask(with_mask)
        loader_test.set_dataset_path(os.path.join(data_path,"easy_pbr_renders"))
        if low_res:
            loader_test.set_subsample_factor(4)
        loader_test.start()
    elif dataset_name=="multiface":
        subject_id=int(scene_name) #a bit hacky but now scene name should actually be an int saying the subject id
        loader_train=DataLoaderMultiFace(config_path, subject_id)
        loader_train.set_dataset_path(os.path.join(data_path,"multiface/multiface_data"))
        if low_res:
            loader_train.set_subsample_factor(4)
        loader_train.set_mode_train()
        loader_train.start()
        #easypbr test
        loader_test=DataLoaderMultiFace(config_path, subject_id)
        loader_test.set_mode_test()
        loader_test.set_dataset_path(os.path.join(data_path,"multiface/multiface_data"))
        if low_res:
            loader_test.set_subsample_factor(4)
        loader_test.start()
    elif dataset_name=="phenorobcp1":
        loader_train=DataLoaderPhenorobCP1(config_path)
        if low_res:
            loader_train.set_subsample_factor(4)
        loader_train.start()
        loader_test=DataLoaderPhenorobCP1(config_path)
        if low_res:
            loader_test.set_subsample_factor(4)
        loader_test.start()
    elif dataset_name=="dtu":
        loader_train=DataLoaderDTU(config_path)
        loader_train.set_dataset_path(os.path.join(data_path,"neus_data/data_DTU"))
        if low_res:
            loader_train.set_subsample_factor(4)
        loader_train.set_mode_train()
        loader_train.set_load_mask(with_mask)
        if scene_name:
            loader_train.set_restrict_to_scene_name(scene_name)
        loader_train.start()
        #the test one has the same scene as the train one 
        loader_test=DataLoaderDTU(config_path)
        loader_test.set_dataset_path(os.path.join(data_path,"neus_data/data_DTU"))
        if low_res:
            loader_test.set_subsample_factor(4)
        loader_test.set_mode_test()
        loader_test.set_load_mask(with_mask)
        if scene_name:
            loader_test.set_restrict_to_scene_name(scene_name)
        loader_test.start()
    elif dataset_name=="bmvs":
        loader_train=DataLoaderDTU(config_path)
        loader_train.set_mode_train()
        loader_train.set_load_mask(with_mask)
        loader_train.set_dataset_path(os.path.join(data_path,"neus_data/data_BlendedMVS"))
        if low_res: 
            loader_train.set_subsample_factor(4)
        if scene_name:
            loader_train.set_restrict_to_scene_name(scene_name)
        loader_train.start()
        #the test one has the same scene as the train one 
        loader_test=DataLoaderDTU(config_path)
        loader_test.set_mode_test()
        loader_test.set_load_mask(with_mask)
        loader_test.set_dataset_path(os.path.join(data_path,"neus_data/data_BlendedMVS"))
        if low_res:
            loader_test.set_subsample_factor(4)
        if scene_name:
            loader_test.set_restrict_to_scene_name(scene_name)
        loader_test.start()
    else:
        print("UNKOWN datasetname in create_dataloader")
        exit(1)

    return loader_train, loader_test

def create_bb_for_dataset(dataset_name):
    if dataset_name=="easypbr":
        aabb=Sphere(0.5, [0,0,0])
    elif dataset_name=="multiface":
        aabb=Sphere(0.5, [0,0,0])
    elif dataset_name=="phenorobcp1":
        #for presenting udirng the workshop at cka
        # aabb=Sphere(0.7, [0,0,0])
        #for the cp1 paper
        aabb=Sphere(0.5, [0,0,0])
    elif dataset_name=="dtu":
        aabb=Sphere(0.5, [0,0,0]) #so we make sure we don't acces the occupancy grid outside
    elif dataset_name=="bmvs":
        aabb=Sphere(0.5, [0,0,0])
    else:
        print("UNKOWN datasetname in create_bb_for_dataset")
        exit(1)

    return aabb

def create_bb_mesh(aabb):

    from easypbr  import Mesh

    if isinstance(aabb, AABB):
        bb_mesh=Mesh()
        bb_mesh.create_box(aabb.bounding_box_sizes_xyz[0], aabb.bounding_box_sizes_xyz[1], aabb.bounding_box_sizes_xyz[2])
        bb_mesh.translate_model_matrix(aabb.bounding_box_translation)
        bb_mesh.m_vis.m_show_mesh=False
        bb_mesh.m_vis.m_show_wireframe=True
    else:
        bb_mesh=Mesh()
        # bb_mesh.create_sphere(aabb.sphere_center, aabb.sphere_radius)
        bb_mesh.create_sphere(aabb.m_center, aabb.m_radius)
        bb_mesh.m_vis.m_show_mesh=False
        bb_mesh.m_vis.m_show_wireframe=True
    return bb_mesh


    
def linear2color_corr(img, dim: int = -1):
    """Applies ad-hoc 'color correction' to a linear RGB Mugsy image along
    color channel `dim` and returns the gamma-corrected result."""

    if dim == -1:
        dim = len(img.shape) - 1

    gamma = 2.0
    black = 3.0 / 255.0
    color_scale = [1.4, 1.1, 1.6]

    assert img.shape[dim] == 3
    if dim == -1:
        dim = len(img.shape) - 1
    scale = np.array(color_scale).reshape([3 if i == dim else 1 for i in range(img.ndim)])
    img = img * scale / 1.1
    return np.clip(
        (((1.0 / (1 - black)) * 0.95 * np.clip(img - black, 0, 2)) ** (1.0 / gamma))
        - 15.0 / 255.0,
        0,
        2,
    )