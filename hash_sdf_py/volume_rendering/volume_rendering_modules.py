import torch
from torch.autograd import Function
from torch import Tensor
from torch.nn import functional as F

import sys
# from instant_ngp_2  import HashTable
from hash_sdf  import VolumeRendering
import numpy as np
import time
import math
from hash_sdf_py.volume_rendering.volume_rendering_funcs import *




class VolumeRenderingGeneralModule(torch.nn.Module):
    def __init__(self):
        super(VolumeRenderingGeneralModule, self).__init__()

    def volume_render_nerf(self, ray_samples_packed, rgb_samples, radiance_samples, ray_t_exit, use_ray_t_exit):

        pred_rgb, pred_depth, bg_transmittance, weight_per_sample = VolumeRenderNerfFunc.apply(ray_samples_packed, rgb_samples, radiance_samples, ray_t_exit, use_ray_t_exit)

        return pred_rgb, pred_depth, bg_transmittance, weight_per_sample

class CumprodAlpha2TransmittanceModule(torch.nn.Module):
    def __init__(self):
        super(CumprodAlpha2TransmittanceModule, self).__init__()

    def forward(self, ray_samples_packed, alpha):

        transmittance, bg_transmittance = CumprodAlpha2TransmittanceFunc.apply(ray_samples_packed, alpha)

        return transmittance, bg_transmittance 


class IntegrateColorAndWeightsModule(torch.nn.Module):
    def __init__(self):
        super(IntegrateColorAndWeightsModule, self).__init__()

    def forward(self, ray_samples_packed, rgb_samples, weights_samples):

        pred_rgb=IntegrateColorAndWeightsFunc.apply(ray_samples_packed, rgb_samples, weights_samples)

        return pred_rgb


class SumOverRayModule(torch.nn.Module):
    def __init__(self):
        super(SumOverRayModule, self).__init__()

    def forward(self, ray_samples_packed, sample_values):

        # weights_sum, weight_sum_per_sample=VolumeRendering.sum_over_each_ray(ray_samples_packed, weights)
        values_sum_per_ray, values_sum_per_sample=SumOverRayFunc.apply(ray_samples_packed, sample_values)

        return values_sum_per_ray, values_sum_per_sample


