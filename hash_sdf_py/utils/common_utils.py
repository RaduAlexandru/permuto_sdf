import torch
import torch.nn as nn
import torch.nn.functional as F
# from easypbr  import *
# from dataloaders import *
import sys
import math
import functools
from matplotlib import cm
import random

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

# from instant_ngp_2_py.utils.aabb import *
# from instant_ngp_2_py.utils.sphere import *
from hash_sdf  import Sphere


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



####from CARE-> utils->math.py

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
    

















def check_args_shadowing(name, method, arg_names):
    spec = inspect.getfullargspec(method)
    init_args = {*spec.args, *spec.kwonlyargs}
    for arg_name in arg_names:
        if arg_name in init_args:
            raise TypeError(f"{name} attempted to shadow a wrapped argument: {arg_name}")


# For backward compatibility.
class TensorMappingHook(object):
    def __init__(
        self,
        name_mapping: List[Tuple[str, str]],
        expected_shape: Optional[Dict[str, List[int]]] = None,
    ):
        """This hook is expected to be used with "_register_load_state_dict_pre_hook" to
        modify names and tensor shapes in the loaded state dictionary.
        Args:
            name_mapping: list of string tuples
            A list of tuples containing expected names from the state dict and names expected
            by the module.
            expected_shape: dict
            A mapping from parameter names to expected tensor shapes.
        """
        self.name_mapping = name_mapping
        self.expected_shape = expected_shape if expected_shape is not None else {}

    def __call__(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        for old_name, new_name in self.name_mapping:
            if prefix + old_name in state_dict:
                tensor = state_dict.pop(prefix + old_name)
                if new_name in self.expected_shape:
                    tensor = tensor.view(*self.expected_shape[new_name])
                state_dict[prefix + new_name] = tensor


def weight_norm_wrapper(cls, name="weight", g_dim=0, v_dim=0):
    """Wraps a torch.nn.Module class to support weight normalization. The wrapped class
    is compatible with the fuse/unfuse syntax and is able to load state dict from previous
    implementations.
    Args:
        name: str
        Name of the parameter to apply weight normalization.
        g_dim: int
        Learnable dimension of the magnitude tensor. Set to None or -1 for single scalar magnitude.
        Default values for Linear and Conv2d layers are 0s and for ConvTranspose2d layers are 1s.
        v_dim: int
        Of which dimension of the direction tensor is calutated independently for the norm. Set to
        None or -1 for calculating norm over the entire direction tensor (weight tensor). Default
        values for most of the WN layers are None to preserve the existing behavior.
    """

    class Wrap(cls):
        def __init__(self, *args, name=name, g_dim=g_dim, v_dim=v_dim, **kwargs):
            # Check if the extra arguments are overwriting arguments for the wrapped class
            check_args_shadowing(
                "weight_norm_wrapper", super().__init__, ["name", "g_dim", "v_dim"]
            )
            super().__init__(*args, **kwargs)

            # Sanitize v_dim since we are hacking the built-in utility to support
            # a non-standard WeightNorm implementation.
            if v_dim is None:
                v_dim = -1
            self.weight_norm_args = {"name": name, "g_dim": g_dim, "v_dim": v_dim}
            self.is_fused = True
            self.unfuse()

            # For backward compatibility.
            self._register_load_state_dict_pre_hook(
                TensorMappingHook(
                    [(name, name + "_v"), ("g", name + "_g")],
                    {name + "_g": getattr(self, name + "_g").shape},
                )
            )

        def fuse(self):
            if self.is_fused:
                return
            # Check if the module is frozen.
            param_name = self.weight_norm_args["name"] + "_g"
            if hasattr(self, param_name) and param_name not in self._parameters:
                raise ValueError("Trying to fuse frozen module.")
            remove_weight_norm(self, self.weight_norm_args["name"])
            self.is_fused = True

        def unfuse(self):
            if not self.is_fused:
                return
            # Check if the module is frozen.
            param_name = self.weight_norm_args["name"]
            if hasattr(self, param_name) and param_name not in self._parameters:
                raise ValueError("Trying to unfuse frozen module.")
            wn = WeightNorm.apply(
                self, self.weight_norm_args["name"], self.weight_norm_args["g_dim"]
            )
            # Overwrite the dim property to support mismatched norm calculate for v and g tensor.
            if wn.dim != self.weight_norm_args["v_dim"]:
                wn.dim = self.weight_norm_args["v_dim"]
                # Adjust the norm values.
                weight = getattr(self, self.weight_norm_args["name"] + "_v")
                norm = getattr(self, self.weight_norm_args["name"] + "_g")
                norm.data[:] = th.norm_except_dim(weight, 2, wn.dim)
            self.is_fused = False

        def __deepcopy__(self, memo):
            # Delete derived tensor to avoid deepcopy error.
            if not self.is_fused:
                delattr(self, self.weight_norm_args["name"])

            # Deepcopy.
            cls = self.__class__
            result = cls.__new__(cls)
            memo[id(self)] = result
            for k, v in self.__dict__.items():
                setattr(result, k, copy.deepcopy(v, memo))

            if not self.is_fused:
                setattr(result, self.weight_norm_args["name"], None)
                setattr(self, self.weight_norm_args["name"], None)
            return result

    return Wrap

def is_weight_norm_wrapped(module):
    for hook in module._forward_pre_hooks.values():
        if isinstance(hook, WeightNorm):
            return True
    return False

class Conv1dUB(th.nn.Conv1d):
    def __init__(self, in_channels, out_channels, width, *args, bias=True, **kwargs):
        """ Conv2d with untied bias. """
        super().__init__(in_channels, out_channels, *args, bias=False, **kwargs)
        self.bias = th.nn.Parameter(th.zeros(out_channels, width)) if bias else None

    # TODO: remove this method once upgraded to pytorch 1.8
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        # Copied from pt1.8 source code.
        if self.padding_mode != 'zeros':
            input = thf.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            return thf.conv1d(
                input, weight, bias, self.stride, _pair(0), self.dilation, self.groups
            )
        return thf.conv1d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input: Tensor) -> Tensor:
        output = self._conv_forward(input, self.weight, None)
        bias = self.bias
        if bias is not None:
            # Assertion for jit script.
            assert bias is not None
            output = output + bias[None]
        return output


class Conv2dUB(th.nn.Conv2d):
    def __init__(self, in_channels, out_channels, height, width, *args, bias=True, **kwargs):
        """ Conv2d with untied bias. """
        super().__init__(in_channels, out_channels, *args, bias=False, **kwargs)
        self.bias = th.nn.Parameter(th.zeros(out_channels, height, width)) if bias else None

    # TODO: remove this method once upgraded to pytorch 1.8
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        # Copied from pt1.8 source code.
        if self.padding_mode != 'zeros':
            input = thf.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            return thf.conv2d(
                input, weight, bias, self.stride, _pair(0), self.dilation, self.groups
            )
        return thf.conv2d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input: Tensor) -> Tensor:
        output = self._conv_forward(input, self.weight, None)
        bias = self.bias
        if bias is not None:
            # Assertion for jit script.
            assert bias is not None
            output = output + bias[None]
        return output


class ConvTranspose1dUB(th.nn.ConvTranspose1d):
    def __init__(self, in_channels, out_channels, width, *args, bias=True, **kwargs):
        """ ConvTranspose1d with untied bias. """
        super().__init__(in_channels, out_channels, *args, bias=False, **kwargs)
        self.bias = th.nn.Parameter(th.zeros(out_channels, width)) if bias else None

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        # Copied from pt1.8 source code.
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')

        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation
        )

        output = thf.conv_transpose1d(
            input,
            self.weight,
            None,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
        bias = self.bias
        if bias is not None:
            # Assertion for jit script.
            assert bias is not None
            output = output + bias[None]
        return output


class ConvTranspose2dUB(th.nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, height, width, *args, bias=True, **kwargs):
        """ ConvTranspose2d with untied bias. """
        super().__init__(in_channels, out_channels, *args, bias=False, **kwargs)
        self.bias = th.nn.Parameter(th.zeros(out_channels, height, width)) if bias else None

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        # Copied from pt1.8 source code.
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation
        )

        output = thf.conv_transpose2d(
            input,
            self.weight,
            None,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
        bias = self.bias
        if bias is not None:
            # Assertion for jit script.
            assert bias is not None
            output = output + bias[None]
        return output



# Set default g_dim=0 (Conv2d) or 1 (ConvTranspose2d) and v_dim=None to preserve
# the current weight norm behavior.
LinearWN = weight_norm_wrapper(th.nn.Linear, g_dim=0, v_dim=None)
Conv1dWN = weight_norm_wrapper(th.nn.Conv1d, g_dim=0, v_dim=None)
Conv1dWNUB = weight_norm_wrapper(Conv1dUB, g_dim=0, v_dim=None)
Conv2dWN = weight_norm_wrapper(th.nn.Conv2d, g_dim=0, v_dim=None)
Conv2dWNUB = weight_norm_wrapper(Conv2dUB, g_dim=0, v_dim=None)
ConvTranspose1dWN = weight_norm_wrapper(th.nn.ConvTranspose1d, g_dim=1, v_dim=None)
ConvTranspose1dWNUB = weight_norm_wrapper(ConvTranspose1dUB, g_dim=1, v_dim=None)
ConvTranspose2dWN = weight_norm_wrapper(th.nn.ConvTranspose2d, g_dim=1, v_dim=None)
ConvTranspose2dWNUB = weight_norm_wrapper(ConvTranspose2dUB, g_dim=1, v_dim=None)



class GatedConv2dWNSwish(torch.nn.Module):
    def __init__(self, in_channels, out_channels, *args, bias=True, **kwargs):
        super(GatedConv2dWNSwish, self).__init__()

        self.conv=Conv2dWN(in_channels, out_channels, *args, bias=bias, **kwargs )
        self.gated_conv=Conv2dWN(in_channels, out_channels, *args, bias=bias, **kwargs )

        self.swish=th.nn.SiLU()
        self.sigmoid=torch.nn.Sigmoid()

    def forward(self, input: Tensor) -> Tensor:
        conv_out=self.swish(self.conv(input))
        gated_out=self.sigmoid(self.gated_conv(input))

        output=conv_out*gated_out

        return output



class InterpolateHook(object):
    def __init__(self, size=None, scale_factor=None, mode="bilinear"):
        """An object storing options for interpolate function"""
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def __call__(self, module, x):
        assert len(x) == 1, "Module should take only one input for the forward method."
        return thf.interpolate(
            x[0],
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=False,
        )


def interpolate_wrapper(cls):
    """Wraps a torch.nn.Module class and perform additional interpolation on the
    first and only positional input of the forward method.
    """

    class Wrap(cls):
        def __init__(self, *args, size=None, scale_factor=None, mode="bilinear", **kwargs):
            check_args_shadowing(
                "interpolate_wrapper", super().__init__, ["size", "scale_factor", "mode"]
            )
            super().__init__(*args, **kwargs)
            self.register_forward_pre_hook(
                InterpolateHook(size=size, scale_factor=scale_factor, mode=mode)
            )

    return Wrap


UpConv2d = interpolate_wrapper(th.nn.Conv2d)
UpConv2dWN = interpolate_wrapper(Conv2dWN)
UpConv2dWNUB = interpolate_wrapper(Conv2dWNUB)


class GlobalAvgPool(th.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], x.shape[1], -1).mean(dim=2)

class Upsample(th.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return thf.interpolate(x, *self.args, **self.kwargs)


# https://github.com/facebookresearch/mvp/blob/d758f53662e79d7fec885f4dd1a3ee457f7c4b00/models/utils.py#L134
class LinearELR(nn.Module):
    """Linear layer with equalized learning rate from stylegan2"""
    def __init__(self, inch, outch, lrmult=1., norm : Optional[str]=None, act_negative_slope=0.0):
        super(LinearELR, self).__init__()

        # lrmult=1e1

        # compute gain from activation fn
        try:
            # if isinstance(act, nn.LeakyReLU):
            actgain = nn.init.calculate_gain("leaky_relu", act_negative_slope)
            # elif isinstance(act, nn.ReLU):
                # actgain = nn.init.calculate_gain("relu")
            # else:
                # actgain = nn.init.calculate_gain(act)
        except:
            actgain = 1.

        initgain = 1. / math.sqrt(inch)

        self.weight = nn.Parameter(torch.randn(outch, inch) / lrmult)
        self.weightgain = actgain

        if norm == None:
            self.weightgain = self.weightgain * initgain * lrmult

        self.bias = nn.Parameter(torch.full([outch], 0.))

        self.norm : Optional[str] = norm
        # self.act = act

        self.fused = False

    def extra_repr(self):
        return 'inch={}, outch={}, norm={}'.format(
            self.weight.size(1), self.weight.size(0), self.norm
        )

    def getweight(self):
        if self.fused:
            return self.weight
        else:
            weight = self.weight
            if self.norm is not None:
                if self.norm == "demod":
                    weight = F.normalize(weight, dim=1)
            return weight

    def fuse(self):
        if not self.fused:
            with torch.no_grad():
                self.weight.data = self.getweight() * self.weightgain
        self.fused = True

    def forward(self, x):
        if self.fused:
            weight = self.getweight()

            out = torch.addmm(self.bias[None], x, weight.t())
            # if self.act is not None:
                # out = self.act(out)
            return out
        else:
            weight = self.getweight()

            # if self.act is None:
                # out = torch.addmm(self.bias[None], x, weight.t(), alpha=self.weightgain)
                # return out
            # else:
            out = F.linear(x, weight * self.weightgain, bias=self.bias)
                # out = self.act(out)
            return out




def leaky_relu_init(m, negative_slope=0.2):

    #mport here in rder to avoid circular dependency
    # from instant_ngp_2_py.lattice.lattice_modules import ConvLatticeIm2RowModule
    # from instant_ngp_2_py.lattice.lattice_modules import CoarsenLatticeModule 
    # from instant_ngp_2_py.lattice.lattice_modules import FinefyLatticeModule 


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
    # #LATTICE THINGS
    # elif isinstance(m, ConvLatticeIm2RowModule):
    #     print("init ConvLatticeIm2RowModule")
    #     # print("conv lattice weight is ", m.weight.shape)
    #     n1 = m.in_channels
    #     n2 = m.out_channels
    #     filter_extent=m.filter_extent
    #     # print("filter_extent", filter_extent)
    #     # print("n1", n1)
    #     std = gain * np.sqrt(2.0 / ((n1 + n2) * filter_extent))
    #     # return
    # elif isinstance(m, CoarsenLatticeModule):
    #     print("init CoarsenLatticeModule")
    #     n1 = m.in_channels
    #     n2 = m.out_channels
    #     filter_extent=m.filter_extent
    #     filter_extent=filter_extent//8
    #     std = gain * np.sqrt(2.0 / ((n1 + n2) * filter_extent))
    # elif isinstance(m, FinefyLatticeModule):
    #     print("init FinefyLatticeModule")
    #     n1 = m.in_channels
    #     n2 = m.out_channels
    #     filter_extent=m.filter_extent
    #     filter_extent=filter_extent//8
    #     #since coarsen usually hits empty space, the effective extent of it is actually smaller
    #     std = gain * np.sqrt(2.0 / ((n1 + n2) * filter_extent))
    else:
        return

    is_wnw = is_weight_norm_wrapped(m)
    if is_wnw:
        m.fuse()

    m.weight.data.uniform_(-std * np.sqrt(3.0), std * np.sqrt(3.0))
    if m.bias is not None:
        m.bias.data.zero_()

    if isinstance(m, th.nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    if is_wnw:
        m.unfuse()

    # m.weights_initialized=True



def swish_init(m, is_linear, scale=1):

    #mport here in rder to avoid circular dependency
    # from instant_ngp_2_py.lattice.lattice_modules import ConvLatticeIm2RowModule
    # from instant_ngp_2_py.lattice.lattice_modules import CoarsenLatticeModule 
    # from instant_ngp_2_py.lattice.lattice_modules import FinefyLatticeModule 


    # is_wnw = is_weight_norm_wrapped(m)
    # if is_wnw:
    #     m.fuse()
    # if hasattr(m, 'weight'):
    #     torch.nn.init.kaiming_normal_(m.weight)
    # if hasattr(m, 'bias'):
    #     if m.bias is not None:
    #         m.bias.data.zero_()
    # if is_wnw:
    #     m.unfuse()
    # return

    #nromally relu has a gain of sqrt(2)
    #however swish has a gain of sqrt(2.952) as per the paper https://arxiv.org/pdf/1805.08266.pdf
    gain=np.sqrt(2.952)
    # gain=np.sqrt(3.2)
    # gain=np.sqrt(3)
    # gain=np.sqrt(2)
    if is_linear:
        gain=1
        # gain = np.sqrt(2.0 / (1.0 + 1 ** 2))
        # print("is lienar")



    if isinstance(m, th.nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        # std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
        std = gain / np.sqrt( ((n1 ) * ksize))
        # std = gain / np.sqrt( ((n2 ) * ksize))
    elif isinstance(m, th.nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        # std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
        std = gain / np.sqrt( ((n1 ) * ksize))
        # std = gain / np.sqrt( ((n2 ) * ksize))
    # elif isinstance(m, PacConv2d):
    #     print("pac init")
    #     ksize = m.kernel_size[0] * m.kernel_size[1]
    #     n1 = m.in_channels
    #     n2 = m.out_channels

    #     # std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    #     std = gain / np.sqrt( ((n1 ) * ksize))
    #     # std = gain / np.sqrt( ((n2 ) * ksize))
    elif isinstance(m, th.nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // 2
        n1 = m.in_channels
        n2 = m.out_channels

        # std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
        std = gain / np.sqrt( ((n1 ) * ksize))
        # std = gain / np.sqrt( ((n2 ) * ksize))
    elif isinstance(m, th.nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // 4
        n1 = m.in_channels
        n2 = m.out_channels

        # std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
        std = gain / np.sqrt( ((n1) * ksize))
        # std = gain / np.sqrt( ((n2) * ksize))
    elif isinstance(m, th.nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] // 8
        n1 = m.in_channels
        n2 = m.out_channels

        # std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
        std = gain / np.sqrt( ((n1 ) * ksize))
        # std = gain / np.sqrt( ((n2 ) * ksize))
    elif isinstance(m, th.nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        # std = gain * np.sqrt(2.0 / (n1 + n2))
        std = gain / np.sqrt( (n1 ))
        # std = gain / np.sqrt( (n2 ))
    #LATTICE THINGS
    elif isinstance(m, ConvLatticeIm2RowModule):
        print("init ConvLatticeIm2RowModule")
        # print("conv lattice weight is ", m.weight.shape)
        n1 = m.in_channels
        n2 = m.out_channels
        filter_extent=m.filter_extent
        # print("filter_extent", filter_extent)
        # print("n1", n1)
        std = gain / np.sqrt( ((n1 ) * filter_extent))
        # return
    elif isinstance(m, CoarsenLatticeModule):
        print("init CoarsenLatticeModule")
        n1 = m.in_channels
        n2 = m.out_channels
        filter_extent=m.filter_extent
        std = gain / np.sqrt( ((n1 ) * filter_extent) *1.0 )
    elif isinstance(m, FinefyLatticeModule):
        print("init FinefyLatticeModule")
        n1 = m.in_channels
        n2 = m.out_channels
        filter_extent=m.filter_extent
        #since coarsen usually hits empty space, the effective extent of it is actually smaller
        std = gain / np.sqrt( ((n1 ) * filter_extent) *0.5 )
        # std = gain / np.sqrt( ((n1 ) * filter_extent) *1.0 )
    else:
        return


    # print("applying init to a ",m)

    is_wnw = is_weight_norm_wrapped(m)
    if is_wnw:
        m.fuse()

    # m.weight.data.uniform_(-std * np.sqrt(3.0), std * np.sqrt(3.0))
    # print("scale is ", scale)
    # print("normal is ", std*scale)
    m.weight.data.normal_(0, std*scale)
    if m.bias is not None:
        m.bias.data.zero_()
        # m.bias.data.normal_(0, np.sqrt(0.04))

    if isinstance(m, th.nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    if is_wnw:
        m.unfuse()

    # m.weights_initialized=True

#init the positional encoding layers, should be done after the swish_init
def pe_layers_init(m):





    if isinstance(m, LearnedPE):
        m.init_weights()
        print("init LearnedPE")
    else:
        return



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


def apply_weight_init_fn_glorot(m, fn):

    should_initialize_weight=True
    if not hasattr(m, "weights_initialized"): #if we don't have this then we need to intiialzie
        # fn(m, is_linear, scale)
        should_initialize_weight=True
    elif m.weights_initialized==False: #if we have it but it's set to false
        # fn(m, is_linear, scale)
        should_initialize_weight=True
    else:
        # print("skipping weight init on ", m)
        should_initialize_weight=False

    if should_initialize_weight:
        fn(m)
        for module in m.children():
            apply_weight_init_fn_glorot(module, fn)




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


#from vol sdf, may need to be deleted
def get_sphere_intersections(cam_loc, ray_directions, r = 1.0):
    # Input: n_rays x 3 ; n_rays x 3
    # Output: n_rays x 1, n_rays x 1 (close and far)

    ray_cam_dot = torch.bmm(ray_directions.view(-1, 1, 3),
                            cam_loc.view(-1, 3, 1)).squeeze(-1)
    under_sqrt = ray_cam_dot ** 2 - (cam_loc.norm(2, 1, keepdim=True) ** 2 - r ** 2)

    # sanity check
    if (under_sqrt <= 0).sum() > 0:
        print('BOUNDING SPHERE PROBLEM!')
        exit()

    sphere_intersections = torch.sqrt(under_sqrt) * torch.Tensor([-1, 1]).cuda().float() - ray_cam_dot
    sphere_intersections = sphere_intersections.clamp_min(0.0)

    return sphere_intersections


def get_combinations(input_tensor):
    # https://discuss.pytorch.org/t/create-all-possible-combinations-of-a-3d-tensor-along-the-dimension-number-1/48155

    #we assume input_tensor is nr_batches, nr_elements, d   and the output is nr_batches ,nr_elements*2, 2d   so it's the combination of all 2 pairs of elements
    assert input_tensor.dim()==3, "We assume that the tensor has 3 dimensions of nr_batches, nr_elements, d"

    first = input_tensor.repeat(1, 2, 1)
    second = input_tensor.unsqueeze(2)
    second = second.repeat(1,1,2,1).view(input_tensor.size(0),-1,input_tensor.size(2))
    output_tensor = torch.cat((first,second), dim=2)

    return output_tensor


def create_dataloader(dataset_name, scene_name, config_path, use_home_computer, use_all_imgs, without_mask):

    from dataloaders import DataLoaderEasyPBR
    from dataloaders import DataLoaderMultiFace
    from dataloaders import DataLoaderPhenorobCP1
    from dataloaders import DataLoaderDTU
    
    if dataset_name=="easypbr":
        # easypbr
        loader_train=DataLoaderEasyPBR(config_path)
        loader_train.set_mode_train()
        loader_train.set_limit_to_nr_imgs(-1)
        loader_train.set_load_mask(not without_mask)
        if use_home_computer:
            loader_train.set_dataset_path("/media/rosu/Data/data/easy_pbr_renders")
        else:
            loader_train.set_dataset_path("/home/user/rosu/data/easy_pbr_renders")
            loader_train.set_subsample_factor(1)
        # loader_train.set_limit_to_nr_imgs(14)
        loader_train.start()
        #easypbr test
        loader_test=DataLoaderEasyPBR(config_path)
        loader_test.set_mode_test()
        loader_test.set_limit_to_nr_imgs(10)
        loader_test.set_load_mask(not without_mask)
        if use_home_computer:
            loader_test.set_dataset_path("/media/rosu/Data/data/easy_pbr_renders")
        else:
            loader_test.set_dataset_path("/home/user/rosu/data/easy_pbr_renders")
            loader_test.set_subsample_factor(1)
        loader_test.start()
    elif dataset_name=="multiface":
        subject_id=int(scene_name) #a bit hacky but now scene name should actually be an int saying the subject id
        loader_train=DataLoaderMultiFace(config_path, subject_id)
        if use_home_computer:
            loader_train.set_dataset_path("/media/rosu/Data/data/multiface/multiface_data")
        else:
            loader_train.set_dataset_path("/home/user/rosu/data/multiface/multiface_data")
            loader_train.set_subsample_factor(1)
        loader_train.set_mode_train()
        if not use_home_computer: #we are on remote so we use the higher resolution
            loader_train.set_subsample_factor(1)
        loader_train.start()
        #easypbr test
        loader_test=DataLoaderMultiFace(config_path, subject_id)
        if use_home_computer:
            loader_test.set_dataset_path("/media/rosu/Data/data/multiface/multiface_data")
        else:
            loader_test.set_dataset_path("/home/user/rosu/data/multiface/multiface_data")
            loader_test.set_subsample_factor(1)
        loader_test.set_mode_test()
        loader_test.start()
    elif dataset_name=="phenorobcp1":
        loader_train=DataLoaderPhenorobCP1(config_path)
        if not use_home_computer: #we are on remote so we use the higher resolution
            loader_train.set_subsample_factor(1)
        loader_train.start()
        loader_test=DataLoaderPhenorobCP1(config_path)
        if not use_home_computer: #we are on remote so we use the higher resolution
            loader_test.set_subsample_factor(1)
        loader_test.start()
    elif dataset_name=="dtu":
        loader_train=DataLoaderDTU(config_path)
        if use_home_computer:
            loader_train.set_dataset_path("/media/rosu/Data/data/neus_data/data_DTU")
        else:
            loader_train.set_dataset_path("/home/user/rosu/data/neus_data/data_DTU")
            loader_train.set_subsample_factor(1)
        loader_train.set_mode_train()
        if use_all_imgs:
            loader_train.set_mode_all()
        loader_train.set_load_mask(not without_mask)


        if scene_name:
            loader_train.set_restrict_to_scene_name(scene_name)
        #set the gpu preloading
        # if use_home_computer:
            # loader_train.set_preload_to_gpu_tensors(False) 
        # else:
            # loader_train.set_preload_to_gpu_tensors(True) 
        loader_train.start()


        #the test one has the same scene as the train one 
        loader_test=DataLoaderDTU(config_path)
        if use_home_computer:
            loader_test.set_dataset_path("/media/rosu/Data/data/neus_data/data_DTU")
        else:
            loader_test.set_dataset_path("/home/user/rosu/data/neus_data/data_DTU")
            loader_test.set_subsample_factor(1)
        loader_test.set_mode_test()
        loader_test.set_load_mask(not without_mask)
        if scene_name:
            loader_test.set_restrict_to_scene_name(scene_name)
        #set the gpu preloading
        # if use_home_computer:
            # loader_test.set_preload_to_gpu_tensors(False) 
        # else:
            # loader_test.set_preload_to_gpu_tensors(True) 
        loader_test.start()
    elif dataset_name=="bmvs":
        loader_train=DataLoaderDTU(config_path)
        loader_train.set_mode_train()
        if use_all_imgs:
            loader_train.set_mode_all()
        loader_train.set_load_mask(not without_mask)

        if use_home_computer:
            loader_train.set_dataset_path("/media/rosu/Data/data/neus_data/data_BlendedMVS")
            # loader_train.set_subsample_factor(4)
        else:
            loader_train.set_dataset_path("/home/user/rosu/data/neus_data/data_BlendedMVS")
            loader_train.set_subsample_factor(1)

        if scene_name:
            loader_train.set_restrict_to_scene_name(scene_name)
        #set the gpu preloading
        # if use_home_computer:
            # loader_train.set_preload_to_gpu_tensors(False) 
        # else:
            # loader_train.set_preload_to_gpu_tensors(True) 
        # loader_train.set_rotate_scene_x_axis_degrees(90)
        # loader_train.set_scene_scale_multiplier(0.5)
        # loader_train.set_restrict_to_scene_name("bmvs_stone")
        loader_train.start()
        #the test one has the same scene as the train one 
        loader_test=DataLoaderDTU(config_path)
        loader_test.set_mode_test()
        loader_test.set_load_mask(not without_mask)
        if use_home_computer:
            loader_test.set_dataset_path("/media/rosu/Data/data/neus_data/data_BlendedMVS")
            # loader_test.set_subsample_factor(4)
        else:
            loader_test.set_dataset_path("/home/user/rosu/data/neus_data/data_BlendedMVS")
            loader_test.set_subsample_factor(1)
        if scene_name:
            loader_test.set_restrict_to_scene_name(scene_name)
        #set the gpu preloading
        # if use_home_computer:
            # loader_test.set_preload_to_gpu_tensors(False) 
        # else:
            # loader_test.set_preload_to_gpu_tensors(True) 
        # loader_test.set_rotate_scene_x_axis_degrees(90)
        # loader_test.set_scene_scale_multiplier(0.5)
        # loader_test.set_restrict_to_scene_name("bmvs_stone")
        loader_test.start()
    elif dataset_name=="nerf":
        loader_train=DataLoaderNerf(config_path)
        if use_home_computer:
            loader_train.set_dataset_path("/media/rosu/Data/data/nerf/nerf_synthetic/nerf_synthetic")
        else:
            loader_train.set_dataset_path("/home/user/rosu/data/neus_data/data_DTU")
        loader_train.set_mode_train()
        if use_all_imgs:
            # loader_train.set_mode_all()
            print("using the train for nerf because all imgs is just too much")
            loader_train.set_mode_train()
        loader_train.set_load_mask(not without_mask)


        if scene_name:
            loader_train.set_restrict_to_scene_name(scene_name)
        #set the gpu preloading
        # if use_home_computer:
            # loader_train.set_preload_to_gpu_tensors(False) 
        # else:
            # loader_train.set_preload_to_gpu_tensors(True) 
        loader_train.start()


        #the test one has the same scene as the train one 
        loader_test=DataLoaderDTU(config_path)
        if use_home_computer:
            loader_test.set_dataset_path("/media/rosu/Data/data/neus_data/data_DTU")
        else:
            loader_test.set_dataset_path("/home/user/rosu/data/neus_data/data_DTU")
        loader_test.set_mode_test()
        loader_test.set_load_mask(not without_mask)
        if scene_name:
            loader_test.set_restrict_to_scene_name(scene_name)
        #set the gpu preloading
        # if use_home_computer:
            # loader_test.set_preload_to_gpu_tensors(False) 
        # else:
            # loader_test.set_preload_to_gpu_tensors(True) 
        loader_test.start()
    else:
        print("UNKOWN datasetname in create_dataloader")
        exit(1)

    return loader_train, loader_test

def create_bb_for_dataset(dataset_name):
    if dataset_name=="easypbr":
        # aabb=AABB(bounding_box_sizes_xyz=[1.0, 1.0, 1.0], bounding_box_translation=[0.0, 0.0, 0.0])
        # aabb_big=AABB(bounding_box_sizes_xyz=[1.0, 1.0, 1.0], bounding_box_translation=[0.0, 0.0, 0.0])
        aabb=Sphere(0.5, [0,0,0])
        aabb_big=Sphere(0.5, [0,0,0])
    elif dataset_name=="multiface":
        # aabb=AABB(bounding_box_sizes_xyz=[1.0, 1.0, 1.0], bounding_box_translation=[0.0, 0.0, 0.0])
        # aabb_big=AABB(bounding_box_sizes_xyz=[1.0, 1.0, 1.0], bounding_box_translation=[0.0, 0.0, 0.0])
        aabb=Sphere(0.5, [0,0,0])
        aabb_big=Sphere(0.5, [0,0,0])
    elif dataset_name=="phenorobcp1":
        # aabb=AABB(bounding_box_sizes_xyz=[1.0, 0.35, 1.0], bounding_box_translation=[0.0, 0.0, 0.0])
        # aabb_big=AABB(bounding_box_sizes_xyz=[1.5, 0.35, 1.5], bounding_box_translation=[0.0, 0.0, 0.0])
        #for maiz
        # aabb=AABB(bounding_box_sizes_xyz=[1.0, 0.8, 1.0], bounding_box_translation=[0.0, 0.09, 0.0])
        # aabb_big=AABB(bounding_box_sizes_xyz=[1.8, 0.8, 1.8], bounding_box_translation=[0.0, 0.09, 0.0])
        #still for maiz but the box is unit cube
        # aabb=AABB(bounding_box_sizes_xyz=[1.0, 1.0, 1.0], bounding_box_translation=[0.0, 0.0, 0.0])
        # aabb_big=AABB(bounding_box_sizes_xyz=[1.8, 1.0, 1.8], bounding_box_translation=[0.0, 0.0, 0.0])

        #for presenting udirng the workshop at cka
        # aabb=Sphere(0.7, [0,0,0])
        # aabb_big=Sphere(0.7, [0,0,0])

        #for the cp1 paper
        aabb=Sphere(0.5, [0,0,0])
        aabb_big=Sphere(0.5, [0,0,0])
    elif dataset_name=="dtu":
        # aabb=AABB(bounding_box_sizes_xyz=[1.0, 1.0, 1.0], bounding_box_translation=[0.0, 0.0, 0.0])
        # aabb_big=AABB(bounding_box_sizes_xyz=[1.0, 1.0, 1.0], bounding_box_translation=[0.0, 0.0, 0.0])
        #better to make ti a sphere because it's easier to creat the background samples for modelling the rest of the scene
        # aabb=SpherePy(radius=0.7, center=[0,0,0])
        # aabb_big=SpherePy(radius=0.7, center=[0,0,0])
        aabb=Sphere(0.49, [0,0,0]) #so we make sure we don't acces the occupancy grid outside
        aabb_big=Sphere(0.49, [0,0,0])
    elif dataset_name=="bmvs":
        # aabb=SpherePy(radius=0.5, center=[0,0,0])
        # aabb_big=SpherePy(radius=0.5, center=[0,0,0])
        radius=0.49
        # if scene_name=="bmvs_dog":
            # radius=0.4 #needs a smaller radius otherwise the dog get learned as images on the backside of the sphere

        aabb=Sphere(radius, [0,0,0])
        aabb_big=Sphere(radius, [0,0,0])
    elif dataset_name=="nerf":
        # aabb=SpherePy(radius=0.5, center=[0,0,0])
        # aabb_big=SpherePy(radius=0.5, center=[0,0,0])
        aabb=Sphere(0.49, [0,0,0])
        aabb_big=Sphere(0.49, [0,0,0])
    
        
    else:
        print("UNKOWN datasetname in create_bb_for_dataset")
        exit(1)

    return aabb, aabb_big

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

# def get_input(phase):
#     gt_mask=None

#      #####GET EAYSPBR
#     if isinstance(phase.loader, DataLoaderEasyPBR):
#         frame=phase.loader.get_next_frame()
#         gt_rgb=mat2tensor(frame.rgb_32f, True).to("cuda")
#     #####GET PHENOROB preloaded
#     elif isinstance(phase.loader, DataLoaderPhenorobCP1):
#         frame_py=random.choice(phase.frames)
#         frame=frame_py.frame
#         gt_rgb=frame_py.rgb_tensor
#         # gt_mask=torch.ones_like(gt_rgb)
#         #get the mask as only 1 channel
#         # gt_mask=gt_mask[:,0:1, :,:]
#     elif isinstance(phase.loader, DataLoaderDTU):
#         frame=phase.loader.get_random_frame()
#         if frame.has_extra_field("has_gpu_tensors"):
#             gt_rgb=frame.get_extra_field_tensor("rgb_32f_tensor")
#             if frame.has_extra_field("mask_tensor"):
#                 gt_mask=frame.get_extra_field_tensor("mask_tensor")
#             else:
#                 gt_mask=None
#         else:
#             gt_rgb=mat2tensor(frame.rgb_32f, True).to("cuda")
#             if not frame.mask.empty():
#                 gt_mask=mat2tensor(frame.mask, False).to("cuda")
#                 gt_rgb=gt_rgb*gt_mask
#                 #get the mask as only 1 channel
#                 gt_mask=gt_mask[:,0:1, :,:]
#     else:
#         print("UNKOWN datasetname in get_input")
#         exit(1)
    
#     return frame, gt_rgb, gt_mask

def define_nr_rays(loader, use_home_computer):

    from dataloaders import DataLoaderEasyPBR
    from dataloaders import DataLoaderMultiFace
    from dataloaders import DataLoaderPhenorobCP1
    from dataloaders import DataLoaderDTU

    if isinstance(loader, DataLoaderEasyPBR):
        # nr_rays=512
        if use_home_computer:
            nr_rays=512
        else:
            nr_rays=512
    elif isinstance(loader, DataLoaderMultiFace):
        # nr_rays=512
        if use_home_computer:
            nr_rays=512
        else:
            nr_rays=512
    elif isinstance(loader, DataLoaderPhenorobCP1):
        # nr_rays=512
        if use_home_computer:
            nr_rays=512
        else:
            nr_rays=512
    elif isinstance(loader, DataLoaderDTU):
        # nr_rays=512
        if use_home_computer:
            nr_rays=512
        else:
            nr_rays=512
    elif isinstance(loader, DataLoaderNerf):
        # nr_rays=512
        if use_home_computer:
            nr_rays=512
        else:
            nr_rays=512
        
    else:
        print("UNKOWN datasetname")
        exit(1)

    return nr_rays 

    
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