import torch
from torch.autograd import Function
from torch import Tensor

from permuto_sdf  import VolumeRendering
import numpy as np
import time
import math

        



class VolumeRenderNerfFunc(Function):
    @staticmethod
    def forward(ctx, ray_samples_packed, rgb_samples, radiance_samples, ray_t_exit, use_ray_t_exit):

        pred_rgb, pred_depth, bg_transmittance, weight_per_sample= VolumeRendering.volume_render_nerf(ray_samples_packed, rgb_samples, radiance_samples, ray_t_exit, use_ray_t_exit) 
        
        ctx.save_for_backward(pred_rgb, rgb_samples, radiance_samples, ray_t_exit, bg_transmittance, weight_per_sample)
        ctx.ray_samples_packed=ray_samples_packed
        ctx.use_ray_t_exit=use_ray_t_exit


        return pred_rgb, pred_depth, bg_transmittance, weight_per_sample



       
    @staticmethod
    def backward(ctx, grad_pred_rgb, grad_depth, grad_bg_transmittance, grad_weight_per_sample):

        pred_rgb, rgb_samples, radiance_samples, ray_t_exit, bg_transmittance, weight_per_sample=ctx.saved_tensors
        ray_samples_packed=ctx.ray_samples_packed
        use_ray_t_exit=ctx.use_ray_t_exit
        
        # grad_rgb_samples=None
        # grad_radiance_samples=None

        # print("grad_depth",grad_depth)
        # print("grad_bg_transmittance",grad_bg_transmittance)

        grad_rgb_samples, grad_radiance_samples =VolumeRendering.volume_render_nerf_backward(grad_pred_rgb, grad_bg_transmittance, grad_weight_per_sample, pred_rgb, ray_samples_packed, rgb_samples, radiance_samples, ray_t_exit, use_ray_t_exit, bg_transmittance) 

        # print("grad_pred_rgb min max", grad_pred_rgb.min(), grad_pred_rgb.max())
        # print("grad_rgb_samples min max", grad_rgb_samples.min(), grad_rgb_samples.max())
        # print("grad_radiance_samples min max", grad_radiance_samples.min(), grad_radiance_samples.max())
      

        ctx.ray_samples_packed=None #Release memory in case it's not automatically released

        return None, grad_rgb_samples, grad_radiance_samples, None, None


class CumprodAlpha2TransmittanceFunc(Function):
    @staticmethod
    def forward(ctx, ray_samples_packed, alpha):

        transmittance, bg_transmittance= VolumeRendering.cumprod_alpha2transmittance(ray_samples_packed, alpha)
        
        ctx.save_for_backward(alpha, transmittance, bg_transmittance)
        ctx.ray_samples_packed=ray_samples_packed


        return transmittance, bg_transmittance



       
    @staticmethod
    def backward(ctx, grad_transmittance, grad_bg_transmittance):

        alpha, transmittance, bg_transmittance=ctx.saved_tensors
        ray_samples_packed=ctx.ray_samples_packed

        # print("grad_bg_transmittance",grad_bg_transmittance.min(), grad_bg_transmittance.max())
        

        #in the forward pass the alphas are a0,a1,a2,a3 
        #the output of cumprod is v0,v1,v2,v3
        #v0=1
        #v1=a0
        #v2=a0*a1
        #v3=a0*a1*a2

        #backward pass we get grad_transmittance which is gradient of loss wrt tranmittnace dL/dV. We call it L 
        #so a vector of L0,L1,L2,L3 
        #the gradient that we should accumulate into alpha in the backwrd pass are 
        #d_a0=L0*0 + L1*1 + L2*a1 + L3*a1*a2
        #d_a1=L2*a0 + L3*a0*a2
        #d_a2=L3*a0*a1
        #d_a3=0 #Dot not get cumprodded in the forward pass so it doesnt matter

        #you can see that this is a sort of cumsum of L*V vectors
        #LV we call C and we have elements C0,C1,C2,C3
        # C0=L0*1
        # C1=L1*a0
        # C2=L2*a0*a1
        # C3=L3*a0*a1*a2 

        #We make a cumsum of them in inverse order which we call P 
        #P0=L3*a0*a1*a2 + L2*a0*a1 + L1*a0 + L0*1
        #P1=L3*a0*a1*a2 + L2*a0*a1 + L1*a0
        #P2=L3*a0*a1*a2 + L2*a0*a1
        #P3=L3*a0*a1*a2

        #Now the gradient we accumulate is just
        # d_a0 = P1/a0


        ###for the bg_transmittance#### 
        #in the forward pass we get bg_transmittance as v3=a0,a1,a2,a3  so a multiplication of all the values except the last alpha a3
        #from the upstream gradient we get dL/dV3
        #we need to push into each alpha the dL\da0, dL\da1, and so on
        # dL\da0 = dL/dV3 * dV3/da0  
        # dV3/da0  = a1*a2 = bg_tranmittance /cur_alpha



        # print("grad_transmittance",grad_transmittance.shape)
        # print("alpha",alpha.shape)
        tensor_LV=grad_transmittance*transmittance
        # tensor_LV=-tensor_LV
        cumsumLV=VolumeRendering.cumsum_over_each_ray(ray_samples_packed, tensor_LV, True)  #the True is for inverse cumsum

        # print("grad_transmittance",grad_transmittance)
        # print("transmittance",transmittance)
        # print("cumsumLV",cumsumLV)
        # print("alpha",alpha)
        # exit(1)

        grad_alpha=None
        grad_alpha =VolumeRendering.cumprod_alpha2transmittance_backward(grad_transmittance, grad_bg_transmittance,ray_samples_packed, alpha, transmittance, bg_transmittance, cumsumLV) 


        ##TODO use als the grad_bg_transmittance

        
        # print("grad_transmittance is ", grad_transmittance.min(), grad_transmittance.max())
        # print("grad_bg_transmittance is ", grad_bg_transmittance.min(), grad_bg_transmittance.max())
        # print("grad_alpha is ", grad_alpha.min(), grad_alpha.max())

        # if torch.isnan(grad_transmittance).any():
        #     print("wtf ")
        #     exit()

        # if torch.isnan(grad_bg_transmittance).any():
        #     print("wtf ")
        #     exit()

        # if torch.isnan(grad_alpha).any():
        #     print("wtf ")
        #     exit()
      

        ctx.ray_samples_packed=None #Release memory in case it's not automatically released

        return None, grad_alpha 


class IntegrateWithWeightsFunc(Function):
    @staticmethod
    def forward(ctx, ray_samples_packed, rgb_samples, weights_samples):

        pred_rgb=VolumeRendering.integrate_with_weights(ray_samples_packed, rgb_samples, weights_samples)
        
        ctx.save_for_backward(rgb_samples, weights_samples, pred_rgb)
        ctx.ray_samples_packed=ray_samples_packed


        return pred_rgb



       
    @staticmethod
    def backward(ctx, grad_pred_rgb):

        rgb_samples, weights_samples, pred_rgb=ctx.saved_tensors
        ray_samples_packed=ctx.ray_samples_packed
        



        grad_rgb_samples, grad_weights_samples =VolumeRendering.integrate_with_weights_backward(grad_pred_rgb, ray_samples_packed,  rgb_samples, weights_samples, pred_rgb) 



        ctx.ray_samples_packed=None #Release memory in case it's not automatically released

        return None, grad_rgb_samples, grad_weights_samples



class SumOverRayFunc(Function):
    @staticmethod
    def forward(ctx, ray_samples_packed, sample_values):

        values_sum_per_ray, values_sum_per_sample=VolumeRendering.sum_over_each_ray(ray_samples_packed, sample_values)

        
        ctx.save_for_backward(sample_values, values_sum_per_ray, values_sum_per_sample)
        ctx.ray_samples_packed=ray_samples_packed


        return values_sum_per_ray, values_sum_per_sample



       
    @staticmethod
    def backward(ctx, grad_values_sum_per_ray, grad_values_sum_per_sample):

        sample_values, values_sum_per_ray, values_sum_per_sample=ctx.saved_tensors
        ray_samples_packed=ctx.ray_samples_packed
        

        grad_sample_values =VolumeRendering.sum_over_each_ray_backward(grad_values_sum_per_ray, grad_values_sum_per_sample, ray_samples_packed, sample_values) 


        ctx.ray_samples_packed=None #Release memory in case it's not automatically released

        return None, grad_sample_values
