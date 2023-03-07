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



#cannot be used with the mask loss because it does not propagate the gradients
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


#volumes renders nerf and can propagate gradients also from the bg_transmittance so it can be used with the mask loss
class VolumeRenderingNerf(torch.nn.Module):
    def __init__(self, ):
        super().__init__()

        self.softplus=torch.nn.Softplus()

        self.cumprod_alpha2transmittance_module=CumprodAlpha2TransmittanceModule()
        self.integrator_module=IntegrateColorAndWeightsModule()
        self.sum_ray_module=SumOverRayModule()

    #The fully fusedvolume rendering in VolumeRenderingGeneralModule.volume_render_nerf doesnt propagate gradients from the sampel weight so we cannot use mask supervision,  we do this now with a more pytorch thing so we can propagate gradient also from the mask loss
    def compute_weights(self, ray_samples_packed, density_samples ):
        dt=ray_samples_packed.samples_dt
        alpha = 1.0 - torch.exp(-density_samples * dt)

        transmittance, bg_transmittance= self.cumprod_alpha2transmittance_module(ray_samples_packed, 1-alpha + 1e-7)

        weights = alpha * transmittance
        weights=weights.view(-1,1)
       

        weights_sum, weight_sum_per_sample=self.sum_ray_module(ray_samples_packed, weights)

        return weights, weights_sum, bg_transmittance
    
    def integrate(self, ray_samples_packed, samples_vals, weights):
        integrated=self.integrator_module(ray_samples_packed, samples_vals, weights)

        return integrated





