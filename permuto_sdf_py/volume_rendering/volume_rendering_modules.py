import torch
from torch.autograd import Function
from torch import Tensor
from torch.nn import functional as F

import sys
from permuto_sdf  import VolumeRendering
import numpy as np
import time
import math
from permuto_sdf_py.volume_rendering.volume_rendering_funcs import *



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


class IntegrateWithWeightsModule(torch.nn.Module):
    def __init__(self):
        super(IntegrateWithWeightsModule, self).__init__()

    def forward(self, ray_samples_packed, rgb_samples, weights_samples):

        pred_rgb=IntegrateWithWeightsFunc.apply(ray_samples_packed, rgb_samples, weights_samples)

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
        self.integrator_module=IntegrateWithWeightsModule()
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




class SingleVarianceNetwork(torch.nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', torch.nn.Parameter(torch.tensor(init_val)))
        self.is_variance_forced=False
        self.last_variance=None
        self.tensor_one=torch.tensor(1.0) #so we don't constantly recreate cuda tensors

    def forward(self, forced_variance=None):
        # print("neus variance is ", self.variance)
        if forced_variance!=None:
            self.is_variance_forced=True
            self.last_variance=forced_variance
            return torch.exp(self.tensor_one*forced_variance * 10.0)
        else:
            self.last_variance=self.variance
            return torch.exp(self.variance * 10.0) 
    
    def get_variance_item(self):
        return self.last_variance

class VolumeRenderingNeus(torch.nn.Module):
    def __init__(self):
        super().__init__()

        init_val=0.3

        self.deviation_network = SingleVarianceNetwork(init_val = init_val).cuda()

        self.cumprod_alpha2transmittance_module=CumprodAlpha2TransmittanceModule()
        self.integrator_module=IntegrateWithWeightsModule()
        self.sum_ray_module=SumOverRayModule()

        self.last_inv_s=None

    def compute_weights(self, ray_samples_packed, sdf, gradients, cos_anneal_ratio, forced_variance=None ):

        nr_samples_total=ray_samples_packed.samples_pos.shape[0]
        dists=ray_samples_packed.samples_dt
        


        #single parameter 
        inv_s = self.deviation_network(forced_variance )           # Single parameter
        inv_s=inv_s.clip(1e-6, 1e6)
        self.last_inv_s=inv_s

        true_cos = (ray_samples_packed.samples_dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # https://github.com/Totoro97/NeuS/issues/35
        #useful for the womask setting
        # iter_cos = -(torch.abs(-true_cos))  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)


        transmittance, bg_transmittance= self.cumprod_alpha2transmittance_module(ray_samples_packed, 1-alpha + 1e-7)

       
        weights = alpha * transmittance
        weights=weights.view(-1,1)
        

        weights_sum, weight_sum_per_sample=self.sum_ray_module(ray_samples_packed, weights)

        return weights, weights_sum, bg_transmittance, inv_s

    def integrate(self, ray_samples_packed, samples_vals, weights):
        integrated=self.integrator_module(ray_samples_packed, samples_vals, weights)

        return integrated

    def get_last_inv_s(self):
        return self.last_inv_s.view(-1)



