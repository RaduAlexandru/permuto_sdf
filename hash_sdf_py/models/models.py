import torch

# import sys
# import os
# import warnings
# from termcolor import colored

import numpy as np
import os
import sys



# from easy_pbr_py.lattice.lattice_py import LatticePy
# from easy_pbr_py.lattice.lattice_funcs import *
# from easy_pbr_py.easy_pbr.easy_pbr_modules import *
# from easy_pbr_py.lattice.lattice_modules import *
# from easy_pbr_py.voxel_grid.voxel_grid_modules import *
# from easy_pbr_py.volume_rendering.volume_rendering_modules import *
from hash_sdf_py.volume_rendering.volume_rendering_modules import *
from hash_sdf_py.models.modules import CreateRaysModule
# from easy_pbr_py.easy_pbr.nerf_utils import *
# from easy_pbr_py.easy_pbr.volsdf_modules import *
# from easy_pbr_py.easy_pbr.neus_modules import *
# from easy_pbr_py.easy_pbr.nerf_modules import *
# from easy_pbr_py.utils.sphere import *
from hash_sdf_py.utils.common_utils import map_range_val
from hash_sdf_py.utils.common_utils import leaky_relu_init
from hash_sdf_py.utils.common_utils import apply_weight_init_fn
from hash_sdf import HashSDF
# from easy_pbr_py.utils.common_utils import cosine_easing_window

# from easy_pbr  import InstantNGP

# import torch.jit as jit

# from functools import reduce
# from torch.nn.modules.module import _addindent

# from torchdiffeq import odeint_adjoint as odeint


#for https://github.com/NVlabs/tiny-cuda-nn
# import commentjson as json

#install with   
    #clone from git@github.com:RaduAlexandru/tiny-cuda-nn.git
    #modify include/tiny-cuda-nn/common.h and set half precision to 0
    # cmake . -B build
    # cmake --build build --config RelWithDebInfo -j
    # pip3 install git+https://github.com/RaduAlexandru/tiny-cuda-nn/#subdirectory=bindings/torch
# import tinycudann as tcnn 


import permutohedral_encoding as permuto_enc


class UNet(torch.nn.Module):
    def __init__(self, in_channels, start_channels, nr_downsamples, compression_rate, out_channels, max_nr_channels):
        super(UNet, self).__init__()

        self.nr_downsamples=nr_downsamples

        initial_channels=in_channels

        cur_nr_channels=in_channels
        self.layers_feat_extractor=torch.nn.ModuleList([])
        self.layers_downsample=torch.nn.ModuleList([])
        self.first_conv=Conv2dWN(cur_nr_channels, start_channels, 3, 1, 1)
        cur_nr_channels=start_channels
        self.nr_channels_vertical_connection=[]
        #making them from fine to coarse
        for i in range(nr_downsamples):
            # if i>0:
                # in_channels=cur_nr_channels+initial_channels-3 #dont cocnat also the pixel dirs
            # else:
            in_channels=cur_nr_channels
            # in_channels=cur_nr_channels
            self.layers_feat_extractor.append(  Conv2dWN(in_channels, cur_nr_channels, 3, 1, 1)  )
            self.nr_channels_vertical_connection.append(cur_nr_channels)
            # self.layers_feat_extractor.append(  ResNetConv(out_channels_per_layer[i])  )
            self.layers_downsample.append( Conv2dWN(cur_nr_channels, min (   int(cur_nr_channels*2*compression_rate), max_nr_channels)  , 4, 2, 1) )
            # self.layers_downsample.append( Conv2dWN(cur_nr_channels, int(cur_nr_channels*2*compression_rate), 3, 1, 1) )
            cur_nr_channels=min( int(cur_nr_channels*2*compression_rate), max_nr_channels)
            # print("unet out nr channel after downsample ", i, " is ", cur_nr_channels)


        self.bottleneck=Conv2dWN(cur_nr_channels, cur_nr_channels, 3, 1, 1)


        #upsample from coarse to fine
        self.layers_up_fuse=torch.nn.ModuleList([])
        self.layers_upsample=torch.nn.ModuleList([])
        #with deconv
        # for i in range(nr_downsamples):
        #     self.layers_upsample.append( ConvTranspose2dWN(cur_nr_channels,int(cur_nr_channels/2),4,2,1)  )
        #     # self.layers_upsample.append( UpsampleConv(cur_nr_channels,int(cur_nr_channels/2) )  )
        #     cur_nr_channels=int(cur_nr_channels/2)
        #     new_channels_vert_connection= self.nr_channels_vertical_connection.pop()
        #     #the features get bilinearly upsampled up to this level and then concatenated and passed through this up_fuse_layer
        #     self.layers_up_fuse.append(  Conv2dWN(cur_nr_channels + new_channels_vert_connection, new_channels_vert_connection, 3, 1, 1)  )
        #     cur_nr_channels=new_channels_vert_connection
        #with bilinear
        for i in range(nr_downsamples):
            #the features get bilinearly upsampled up to this level and then concatenated and passed through this up_fuse_layer
            new_channels_vert_connection= self.nr_channels_vertical_connection.pop()
            self.layers_up_fuse.append(  Conv2dWN(cur_nr_channels + new_channels_vert_connection, new_channels_vert_connection, 3, 1, 1)  )
            cur_nr_channels=new_channels_vert_connection

        #add one more conv and swish at the highest resolution to further improve high frquency details
        self.prelast= Conv2dWN(cur_nr_channels, cur_nr_channels, 3, 1, 1)

        #last layer
        self.last= Conv2dWN(cur_nr_channels, out_channels, 1, 1, 0)






        self.activ=th.nn.Mish()
        self.relu=th.nn.ReLU()



        # print("initializing weights for unet")
        apply_weight_init_fn(self, leaky_relu_init, negative_slope=0.0)
        leaky_relu_init(self.last, negative_slope=1.0)
        # exit(1)


    # def forward(self, img_splatted_features, pixel_dirs):
    def forward(self, x):



        #downsample
        x=self.first_conv(x)
        x=self.activ(x)
        feat_per_res=[]
        for i in range(self.nr_downsamples):
            x=self.layers_feat_extractor[i](x)
            x=self.activ(x)
            # print("appending x of size", x.shape)
            feat_per_res.append(x)
            #downsample
            # print("input to downsample ", i, " x is ", x.shape)
            x=self.layers_downsample[i](x)
            x=self.activ(x)
            #do not use the resize because for some reason it shift the image by a few pixels
            # x=resize_right.resize(x, scale_factors=0.5,  interp_method=interp_methods.linear)

        #bottlneck
        # print("input bottlencj x is ", x.shape)
        x=self.bottleneck(x)
        x=self.activ(x)

        #upsample
        for i in range(self.nr_downsamples):
            vert_connection=feat_per_res.pop()
            # print("")
            # print("vert_connection ", i, " vert_connection is ", vert_connection.shape)
            #upsample the current x to be the same size as the vertical connection
            # print("before upsample ", i, " x is ", x.shape)
            x=torch.nn.functional.interpolate(x, size=(vert_connection.shape[2], vert_connection.shape[3]), mode='bilinear', align_corners=False )
            # x=torch.nn.functional.interpolate(x, size=(int(x.shape[2]*2), int(x.shape[3]*2) ), mode='bilinear', align_corners=False )
            # print("before upsample ", i, " x is ", x.shape)
            # x=self.layers_upsample[i](x)
            # x=self.swish(x)
            # print("after upsample ", i, " x is ", x.shape)
            x=torch.cat([x,vert_connection],1)
            # print("after concat ", i, " x is ", x.shape)
            x=self.layers_up_fuse[i](x)
            x=self.activ(x)
            # print("after fuse ", i, " x is ", x.shape)

        x=self.prelast(x)
        x=self.activ(x)

        #last conv
        x=self.last(x)


        return x

class MLP(torch.jit.ScriptModule):

    def __init__(self, in_channels, hidden_dim, out_channels, nr_layers, last_layer_linear_init):
        super(MLP, self).__init__()


        # self.mlp_feat_and_density= nn.Sequential(
        #     torch.nn.Linear(nr_lattice_features*nr_resolutions_to_slice, 64),
        #     torch.nn.Mish(),
        #     torch.nn.Linear(64,64),
        #     torch.nn.Mish(),
        #     torch.nn.Linear(64,64),
        #     torch.nn.Mish(),
        #     torch.nn.Linear(64,self.nr_feat_for_rgb+1) 
        # )
        # apply_weight_init_fn(self.mlp_feat_and_density, leaky_relu_init, negative_slope=0.0)
        # # leaky_relu_init(self.mlp_feat_and_density[-1], negative_slope=1.0)


        self.layers=[]
        self.layers.append(  torch.nn.Linear(in_channels, hidden_dim) )
        self.layers.append(  torch.nn.Mish() )
        for i in range(nr_layers):
            self.layers.append(   torch.nn.Linear(hidden_dim, hidden_dim)  )
            self.layers.append(   torch.nn.Mish()  )
        self.layers.append(  torch.nn.Linear(hidden_dim, out_channels  )   )

        self.mlp=torch.nn.Sequential(*self.layers)

        apply_weight_init_fn(self.mlp, leaky_relu_init, negative_slope=0.0)
        if last_layer_linear_init:
            leaky_relu_init(self.mlp[-1], negative_slope=1.0)

        self.weights_initialized=True #so that apply_weight_init_fn doesnt initialize anything

    @torch.jit.script_method
    def forward(self, x):

        x=self.mlp(x)

        return x

#from https://arxiv.org/pdf/2202.08345.pdf
class LipshitzMLP(torch.nn.Module):

    def __init__(self, in_channels, nr_out_channels_per_layer, last_layer_linear):
        super(LipshitzMLP, self).__init__()


        self.last_layer_linear=last_layer_linear
     

        self.layers=torch.nn.ParameterList()
        # self.layers=[]
        for i in range(len(nr_out_channels_per_layer)):
            cur_out_channels=nr_out_channels_per_layer[i]
            self.layers.append(  torch.nn.Linear(in_channels, cur_out_channels)   )
            in_channels=cur_out_channels
        apply_weight_init_fn(self, leaky_relu_init, negative_slope=0.0)
        if last_layer_linear:
            leaky_relu_init(self.layers[-1], negative_slope=1.0)

        #we make each weight separately because we want to add the normalize to it
        self.weights_per_layer=torch.nn.ParameterList()
        self.biases_per_layer=torch.nn.ParameterList()
        for i in range(len(self.layers)):
            self.weights_per_layer.append( self.layers[i].weight  )
            self.biases_per_layer.append( self.layers[i].bias  )

        self.lipshitz_bound_per_layer=torch.nn.ParameterList()
        for i in range(len(self.layers)):
            max_w= torch.max(torch.sum(torch.abs(self.weights_per_layer[i]), dim=1))
            #we actually make the initial value quite large because we don't want at the beggining to hinder the rgb model in any way. A large c means that the scale will be 1
            c = torch.nn.Parameter(  torch.ones((1))*max_w*2 ) 
            self.lipshitz_bound_per_layer.append(c)






        self.weights_initialized=True #so that apply_weight_init_fn doesnt initialize anything

    def normalization(self, w, softplus_ci):
        absrowsum = torch.sum(torch.abs(w), dim=1)
        scale = torch.minimum(torch.tensor(1.0), softplus_ci/absrowsum)
        return w * scale[:,None]

    def lipshitz_bound_full(self):
        lipshitz_full=1
        for i in range(len(self.layers)):
            lipshitz_full=lipshitz_full*torch.nn.functional.softplus(self.lipshitz_bound_per_layer[i])

        return lipshitz_full

    def forward(self, x):

        # x=self.mlp(x)

        for i in range(len(self.layers)):
            weight=self.weights_per_layer[i]
            bias=self.biases_per_layer[i]

            weight=self.normalization(weight, torch.nn.functional.softplus(self.lipshitz_bound_per_layer[i])  )

            x=torch.nn.functional.linear(x, weight, bias)

            is_last_layer=i==(len(self.layers)-1)

            if is_last_layer and self.last_layer_linear:
                pass
            else:
                x=torch.nn.functional.gelu(x)


        return x


##############################instant NGP things
class InstantNGP2(torch.nn.Module):
    def __init__(self, nr_lattice_vertices, nr_lattice_features, nr_resolutions, min_depth, max_depth, nr_samples_per_ray, bounding_box_sizes_xyz, bounding_box_translation):
        super(InstantNGP2, self).__init__()

        self.nr_resolutions=nr_resolutions
        self.bounding_box_sizes_xyz=bounding_box_sizes_xyz
        self.bounding_box_translation=bounding_box_translation

        self.pick_rand_pixels= RandPixelPicker()
        self.pixel_sampler=PixelSampler()
        self.create_ray_samples=CreateRaySamplesModule(min_depth=min_depth, max_depth=max_depth, nr_samples_per_ray=nr_samples_per_ray)
        self.slice_lattice=SliceLatticeWithCollisionsModule()

   


        self.rgba_lv_delta_list = torch.nn.ParameterList()
        self.sigmas_list=np.linspace(0.1, 0.001, num=nr_resolutions) #the smaller the value, the finer the lattice
        for i in range(nr_resolutions):
            self.rgba_lv_delta_list.append(  torch.nn.Parameter( torch.randn( nr_lattice_vertices, nr_lattice_features  )*1e-3 , requires_grad=True)  )

        self.mlp_rbga=model = nn.Sequential(
            LinearWN(3+9+nr_lattice_features*nr_resolutions,64),
            torch.nn.Softplus(),
            LinearWN(64,32),
            torch.nn.Softplus(),
            LinearWN(32,4),
        )

        apply_weight_init_fn(self, leaky_relu_init, negative_slope=0.0)
        # leaky_relu_init(self.mlp_xyz[-1], 1.0)
        leaky_relu_init(self.mlp_rbga[-1], 1.0)

        self.c2f=Coarse2Fine(self.nr_resolutions)


    def forward(self, ls, frame, ray_samples, ray_dirs, samples_depth, iter_nr):

        nr_rays=ray_samples.shape[0]
        nr_samples_per_ray=ray_samples.shape[1]


        ##deform xyz
        xyz_delta=None


        #coarse to fine
        # window=cosine_easing_window(self.nr_resolutions, iter_nr*10.001)
        window=self.c2f(iter_nr*0.001)
        # print("window", window)

        
        #get rgba
        TIME_START("slice_full")
        ray_features_list=[]
        for i in range(self.nr_resolutions):
            #set the lattice values to zero because we will delta them with the parameters
            ls.set_sigma(self.sigmas_list[i])
            values=self.rgba_lv_delta_list[i] * window[i]
            ray_features=self.slice_lattice(values, ls, ray_samples.view(-1,3) )
            ray_features=ray_features.view(nr_rays, nr_samples_per_ray, -1)
            ray_features_list.append(ray_features)
        ray_features=torch.cat(ray_features_list, 2)
        TIME_END("slice_full")


        #encode rays
        ray_dirs_lin=ray_dirs.view(-1,3)
        ray_dirs_encoded=spherical_harmonics_basis(ray_dirs_lin)
        ray_dirs_encoded=ray_dirs_encoded.view(nr_rays, nr_samples_per_ray, -1)

        x=torch.cat([ray_samples,ray_dirs_encoded,ray_features],2)
        rgb_sigma=self.mlp_rbga(x)
        # rgb_sigma+=0.5

        # print("sigma_rgba", sigma_rgba.shape)
        # print("samples_depth", samples_depth.shape)
        # print("ray_dirs",ray_dirs.shape)
        ray_dirs=ray_dirs[:,0,:]

        #set the radiance of the points outside of the bounding box to zero 
        mask_inside_bb=check_point_inside_cube(ray_samples, self.bounding_box_sizes_xyz, self.bounding_box_translation)*1.0
        rgb_sigma=rgb_sigma*mask_inside_bb

        rgb_map, disp_map, acc_map, weights, depth_map= volume_render_radiance_field( rgb_sigma, samples_depth.squeeze(),  ray_dirs)


        return rgb_map, rgb_sigma, xyz_delta, ray_samples

class InstantNGP2_depth_multiplier(torch.nn.Module):
    def __init__(self, nr_lattice_vertices, nr_lattice_features, nr_resolutions):
        super(InstantNGP2_depth_multiplier, self).__init__()

        self.nr_resolutions=nr_resolutions

        self.pick_rand_pixels= RandPixelPicker()
        self.pixel_sampler=PixelSampler()
        self.create_ray_samples=CreateRaySamplesModule(min_depth=min_depth, max_depth=max_depth, nr_samples_per_ray=nr_samples_per_ray, randomize_depth=True)
        self.slice_lattice=SliceLatticeWithCollisionsModule()



        #features and mlp for depth xyz
        self.xyz_lv_delta_list = torch.nn.ParameterList()
        self.sigmas_list=np.linspace(0.1, 0.01, num=nr_resolutions) #the smaller the value, the finer the lattice
        for i in range(nr_resolutions):
            self.xyz_lv_delta_list.append(  torch.nn.Parameter( torch.randn( nr_lattice_vertices, nr_lattice_features  )*1e-3 , requires_grad=True)  )

        self.mlp_depth_scaling=model = nn.Sequential(
            LinearWN(3+9+nr_lattice_features*nr_resolutions,32),
            torch.nn.LeakyReLU(0.2),
            LinearWN(32,32),
            torch.nn.LeakyReLU(0.2),
            LinearWN(32,1)
        )



       

        apply_weight_init_fn(self, leaky_relu_init, alpha=0.2)
        leaky_relu_init(self.mlp_depth_scaling[-1], 1.0)

        


    def forward(self, ls, frame, ray_samples, ray_dirs, samples_depth):

        nr_rays=ray_samples.shape[0]
        nr_samples_per_ray=ray_samples.shape[1]


        ##deform xyz
        xyz_delta=None
        TIME_START("slice_full")
        ray_features_list=[]
        for i in range(self.nr_resolutions):
            ls.set_sigma(self.sigmas_list[i])
            lv=ls.values()
            lv+=self.xyz_lv_delta_list[i]
            ray_features=self.slice_lattice(lv, ls, ray_samples.view(-1,3) )
            ray_features=ray_features.view(nr_rays, nr_samples_per_ray, -1)
            ray_features_list.append(ray_features)
        ray_features=torch.cat(ray_features_list, 2)
        TIME_END("slice_full")

        #encode rays
        ray_dirs_lin=ray_dirs.view(-1,3)
        ray_dirs_encoded=spherical_harmonics_basis(ray_dirs_lin)
        ray_dirs_encoded=ray_dirs_encoded.view(nr_rays, nr_samples_per_ray, -1)

        x=torch.cat([ray_samples,ray_dirs_encoded,ray_features],2)
        depth_delta=self.mlp_depth_scaling(x) #nr_raysxnr_samplesx3

        depth_delta=1.0 + depth_delta*0.01
        print("depth_delta min max", depth_delta.min(), depth_delta.max())
        depth_delta=torch.nn.functional.relu(depth_delta) #prevent negative depth_multiplier so we cannot flip the ray
       

        return depth_delta

class InstantNGP2ODE(torch.nn.Module):
    def __init__(self, nr_lattice_vertices, nr_lattice_features, nr_resolutions):
        super(InstantNGP2ODE, self).__init__()

        self.nr_resolutions=nr_resolutions
        self.min_depth=min_depth
        self.max_depth=max_depth

        self.pick_rand_pixels= RandPixelPicker()
        self.pixel_sampler=PixelSampler()
        self.create_ray_samples=CreateRaySamplesModule(min_depth=min_depth, max_depth=max_depth, nr_samples_per_ray=nr_samples_per_ray, randomize_depth=True)


        self.ode=ODEFunc(nr_lattice_vertices, nr_lattice_features, nr_resolutions)


       
        


    def forward(self, ls, frame, ray_samples, ray_dirs, samples_depth):

        nr_rays=ray_samples.shape[0]
        nr_samples_per_ray=ray_samples.shape[1]


        ##deform xyz
        xyz_delta=None

        #get features for the ray, like dirs
        ray_dirs=ray_dirs[:,0,:] #nr_rays x 3
        #encode rays
        ray_dirs_lin=ray_dirs.view(-1,3)
        ray_dirs_encoded=spherical_harmonics_basis(ray_dirs_lin)
        ray_dirs_encoded=ray_dirs_encoded.view(nr_rays, -1)
        ray_features=ray_dirs_encoded
       

        # y0 is an any-D Tensor representing the initial values, and t is a 1-D Tensor containing the evaluation points. The initial time is taken to be t[0]
        timesteps_to_sample = torch.tensor([0, 1]).float()
        start_rgba=torch.zeros(nr_rays,4).cuda()
        self.ode.nfe=0
        cam_origin=torch.from_numpy(frame.pos_in_world()).cuda().view(-1,3)
        print("calling solve_ode")
        integrated_rgba= self.ode.solve_ode(timesteps_to_sample, start_rgba, ray_features, ray_dirs, cam_origin, ls, self.min_depth, self.max_depth)
        print("finished solve_ode")


        #return values
        rgb_map=integrated_rgba[:, 0:3]
        rgb_sigma=integrated_rgba
        
        print("rgb_map", rgb_map.shape) 

        return rgb_map, rgb_sigma, xyz_delta, ray_samples

class ODEFunc(torch.nn.Module):

    def __init__(self, nr_lattice_vertices, nr_lattice_features, nr_resolutions):
        super(ODEFunc, self).__init__()

        self.nr_resolutions=nr_resolutions

        self.nfe=0
        self.ray_features=None #this will get set when you call solve_ode since it is constant for all the samples on the ray
        self.ls=None
        
        self.slice_lattice=SliceLatticeWithCollisionsModule()
        self.rgba_lv_delta_list = torch.nn.ParameterList()
        self.sigmas_list=np.linspace(0.1, 0.01, num=nr_resolutions) #the smaller the value, the finer the lattice
        for i in range(nr_resolutions):
            self.rgba_lv_delta_list.append(  torch.nn.Parameter( torch.randn( nr_lattice_vertices, nr_lattice_features  )*1e-3 , requires_grad=True)  )

        self.mlp_rbga=model = nn.Sequential(
            LinearWN(3+9+nr_lattice_features*nr_resolutions,32),
            torch.nn.SiLU(),
            LinearWN(32,32),
            torch.nn.SiLU(),
            LinearWN(32,4),
        )

       


    #this function gets called automatically by the ODE solver so we never have to call it explicity. We only call explicitly the solve_ode function
    #pos is #nr_rays x 3
    def forward(self, cur_t, cur_integrated_rgba):

        self.nfe+=1

        # print("odefunc forward with cur_integrated_rgba", cur_integrated_rgba.shape)
        # print("odefunc forward with cur_t", cur_t.shape)
        print("cur_t is ", cur_t)

        #calculate the ray_samples based on the camera position + ray_dirs* map_range(cur_t, 0, 1, min_depth, max_depth) 
        ray_samples=self.cam_origin + self.ray_dirs * map_range_val(cur_t.item(), 0.0, 1.0, self.min_depth, self.max_depth)
        # print("ray_samples", ray_samples.shape)

        nr_rays=ray_samples.shape[0]

        TIME_START("slice_full")
        sample_features_list=[]
        for i in range(self.nr_resolutions):
            self.ls.set_sigma(self.sigmas_list[i])
            lv=self.ls.values()
            lv+=self.rgba_lv_delta_list[i]
            sample_features=self.slice_lattice(lv, self.ls, ray_samples.view(-1,3) )
            sample_features=sample_features.view(nr_rays, -1)
            sample_features_list.append(sample_features)
        sample_features=torch.cat(sample_features_list, 1)
        TIME_END("slice_full")


        # print("self.ray_features", self.ray_features.shape)        
        # print("sample_features", sample_features.shape)        

        x=torch.cat([ray_samples,self.ray_features,sample_features],1)
        rgba_gradient=self.mlp_rbga(x)
        # rgba_gradient+=0.5
        # rgba_gradient+=0.5
        
        #set that we can only add color, we cannot substract it
        rgba_gradient=rgba_gradient.abs()

        #when the alpha of the accumulation reaches 1.0. the gradient should stop accumulating, the gradient should be zero
        # rgba_gradient[:, 3:4]=torch.exp(rgba_gradient[:, 3:4].clone())
        rgba_gradient[:, 0:3] = rgba_gradient[:, 0:3].clone()*rgba_gradient[:, 3:4].clone() #attenuate by the alpha
        # print("cur_integrated_rgba", cur_integrated_rgba)
        # non_saturated_pixels=(cur_integrated_rgba[:, 3:4].clone()<1.0) #1 for non saturated, zero for saturated pixels
        # rgba_gradient=rgba_gradient.clone()*non_saturated_pixels

        



        # print("rgba_gradient", rgba_gradient.shape)

        # print("rgba_gradient", rgba_gradient)


        return rgba_gradient



    #required in order to pass additional inputs to odeint similar to https://github.com/rtqichen/torchdiffeq/issues/129
    #start_rgb: Nx4
    #ray_features: NxM
    #ray_dirs: Nx3
    #cam_origin: 1x3
    def solve_ode(self, timesteps_to_sample, start_rgba, ray_features, ray_dirs, cam_origin, ls, min_depth, max_depth):
        self.ray_features = ray_features  # overwrite it
        self.ray_dirs=ray_dirs
        self.cam_origin=cam_origin
        self.ls=ls
        self.min_depth=min_depth
        self.max_depth=max_depth
        # outputs = odeint(self, x0, t, ...)

        # outputs = odeint_adjoint(self, initial_pos, timesteps_to_sample, adjoint_params=list(self.parameters()) + [self.strand_features] )
        # outputs = odeint_adjoint(self, initial_pos, timesteps_to_sample, adjoint_params=list(self.parameters()) + [self.strand_features], rtol=1e-4, atol=1e-4)
        # outputs = odeint_adjoint(self, initial_pos, timesteps_to_sample, adjoint_params=list(self.parameters()) + [self.strand_features], method="rk4", options=dict(step_size=0.1) )
        # outputs = odeint(self, start_rgba, timesteps_to_sample, adjoint_params=list(self.parameters()) + [self.ray_features],
        outputs = odeint(self, start_rgba, timesteps_to_sample,
            # method="euler",
            # method="bosh3",
            method="rk4",
            options=dict(step_size=0.1),
            # adjoint_options={
                # "norm": "seminorm"
            #   }
            )

        print("outputs of ode is ", outputs.shape)

        # return outputs[1]
        return outputs[1, :, :] #return the end of the sequence

##############################VOLSDF#################
class ImplicitNetwork(torch.nn.Module):

    def __init__(self, nr_lattice_vertices, nr_lattice_features, nr_resolutions):
        super(ImplicitNetwork, self).__init__()

        self.nr_resolutions=nr_resolutions
        self.slice_lattice=SliceLatticeWithCollisionsModule()
        self.rgba_lv_delta_list = torch.nn.ParameterList()
        self.sigmas_list=np.linspace(0.1, 0.001, num=nr_resolutions) #the smaller the value, the finer the lattice
        for i in range(nr_resolutions):
            self.rgba_lv_delta_list.append(  torch.nn.Parameter( torch.randn( nr_lattice_vertices, nr_lattice_features  )*1e-3 , requires_grad=True)  )

        self.mlp_sdf=model = nn.Sequential(
            LinearWN(3+nr_lattice_features*nr_resolutions,64),
            torch.nn.Softplus(),
            LinearWN(64,32),
            torch.nn.Softplus(),
            LinearWN(32,1),
            torch.nn.Softplus()
        )

        # nr_hidden=64
        # start_nr_channels=3+nr_lattice_features*nr_resolutions
        # self.mlp_sdf_list=torch.nn.ModuleList([])
        # self.mlp_sdf_list.append(   torch.nn.Linear(start_nr_channels, nr_hidden )   )
        # for i in range(2):
        #     self.mlp_sdf_list.append(   torch.nn.Linear(nr_hidden, nr_hidden)   )
        # self.mlp_sdf_list.append(   torch.nn.Linear(nr_hidden, 1)   )

        # #geometric init https://arxiv.org/pdf/1911.10414.pdf
        # p=1.0
        # lin=self.mlp_sdf_list[0]
        # torch.nn.init.normal_(lin.weight, mean=2*np.sqrt(np.pi) / np.sqrt(p * start_nr_channels), std=0.000001)
        # torch.nn.init.constant_(lin.bias, -1.0)
        # for i in range(len(self.mlp_sdf_list)-2):
        #     lin=self.mlp_sdf_list[i+1]
        #     torch.nn.init.normal_(lin.weight, mean=2*np.sqrt(np.pi) / np.sqrt(p * nr_hidden), std=0.000001)
        #     torch.nn.init.constant_(lin.bias, -1.0)
        # lin=self.mlp_sdf_list[-1]
        # torch.nn.init.constant_(lin.bias, 0.0)
        # torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(p*1))

        self.softplus=torch.nn.Softplus()

        apply_weight_init_fn(self, leaky_relu_init, negative_slope=0.0)
        leaky_relu_init(self.mlp_sdf[-1], 1.0)


        self.c2f=Coarse2Fine(self.nr_resolutions)

    
    def forward(self, ray_samples_flat, ls, iter_nr):

        assert ray_samples_flat.shape[1] == 3, "ray_samples_flat should be nx3"

        window=self.c2f(iter_nr*0.0001)

        TIME_START("slice_full")
        ray_features_list=[]
        for i in range(self.nr_resolutions):
            #set the lattice values to zero because we will delta them with the parameters
            ls.set_sigma(self.sigmas_list[i])
            values=self.rgba_lv_delta_list[i] * window[i]
            ray_features=self.slice_lattice(values, ls, ray_samples_flat )
            ray_features_list.append(ray_features)
        ray_features=torch.cat(ray_features_list, 1)
        TIME_END("slice_full")



        x=torch.cat([ray_samples_flat,ray_features],1)
        x=self.mlp_sdf(x)

        # for i in range(len(self.mlp_sdf_list)):
        #     x=self.mlp_sdf_list[i](x)
        #     x=self.softplus(x)


        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def get_outputs(self, x, ls, iter_nr):
        x.requires_grad_(True)
        output = self.forward(x, ls, iter_nr)
        sdf = output[:,:1]
        # ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        # if self.sdf_bounding_sphere > 0.0:
        #     sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
        #     sdf = torch.minimum(sdf, sphere_sdf)
        # feature_vectors = output[:, 1:]

        feature_vectors=None
        # d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        # gradients = torch.autograd.grad(
        #     outputs=sdf,
        #     inputs=x,
        #     grad_outputs=d_output,
        #     create_graph=True,
        #     retain_graph=True,
        #     only_inputs=True)[0]
        gradients=None

        return sdf, feature_vectors, gradients

class RenderingNetwork(torch.nn.Module):
    def __init__(self, nr_lattice_vertices, nr_lattice_features, nr_resolutions ):
        super().__init__()

        self.nr_resolutions=nr_resolutions
        self.slice_lattice=SliceLatticeWithCollisionsModule()
        self.rgba_lv_delta_list = torch.nn.ParameterList()
        self.sigmas_list=np.linspace(0.1, 0.001, num=nr_resolutions) #the smaller the value, the finer the lattice
        for i in range(nr_resolutions):
            self.rgba_lv_delta_list.append(  torch.nn.Parameter( torch.randn( nr_lattice_vertices, nr_lattice_features  )*1e-3 , requires_grad=True)  )

        self.mlp_rgb=model = nn.Sequential(
            LinearWN(3 +nr_lattice_features*nr_resolutions,64),
            torch.nn.Softplus(),
            LinearWN(64,32),
            torch.nn.Softplus(),
            LinearWN(32,3)
        )
        
        self.sigmoid = torch.nn.Sigmoid()

        self.c2f=Coarse2Fine(self.nr_resolutions)


        apply_weight_init_fn(self, leaky_relu_init, negative_slope=0.0)
        leaky_relu_init(self.mlp_rgb[-1], 1.0)

    def forward(self, points, normals, view_dirs, feature_vectors, ls, iter_nr):

        # x=points 

        window=self.c2f(iter_nr*0.0001)

        TIME_START("slice_full")
        ray_features_list=[]
        for i in range(self.nr_resolutions):
            #set the lattice values to zero because we will delta them with the parameters
            ls.set_sigma(self.sigmas_list[i])
            values=self.rgba_lv_delta_list[i] * window[i]
            ray_features=self.slice_lattice(values, ls, points )
            ray_features_list.append(ray_features)
        ray_features=torch.cat(ray_features_list, 1)
        TIME_END("slice_full")

        x=torch.cat([points,ray_features],1)



        x=self.mlp_rgb(x)


        x = self.sigmoid(x)
        return x

class VolSDF(torch.nn.Module):
    def __init__(self, nr_lattice_vertices, nr_lattice_features, nr_resolutions, min_depth, max_depth, nr_samples_per_ray, bounding_box_sizes_xyz, bounding_box_translation):
        super(VolSDF, self).__init__()

        self.nr_resolutions=nr_resolutions
        self.bounding_box_sizes_xyz=bounding_box_sizes_xyz
        self.bounding_box_translation=bounding_box_translation

        self.pick_rand_pixels= RandPixelPicker()
        self.pixel_sampler=PixelSampler()
        self.create_ray_samples=CreateRaySamplesModule(min_depth=min_depth, max_depth=max_depth, nr_samples_per_ray=nr_samples_per_ray)


        self.implicit_network=ImplicitNetwork(nr_lattice_vertices, nr_lattice_features, nr_resolutions)
        self.rendering_network=RenderingNetwork(nr_lattice_vertices, nr_lattice_features, nr_resolutions)
        self.density = LaplaceDensity()



    def forward(self, ls, frame, ray_samples, ray_dirs, samples_depth, iter_nr):

        # print("\nforward--ray_samples", ray_samples.shape)

        nr_rays=ray_samples.shape[0]
        nr_samples_per_ray=ray_samples.shape[1]


        ##deform xyz
        xyz_delta=None




        #run the implicit network https://github.com/lioryariv/volsdf/blob/d861914077cf3e35d5841515b03bf4355cf85373/code/model/network_bg.py#L63
        sdf, feature_vectors, gradients=self.implicit_network.get_outputs(ray_samples.view(-1,3), ls, iter_nr)
        # print("sdf", sdf.shape)


        rgb_flat = self.rendering_network(ray_samples.view(-1,3), gradients, ray_dirs.view(-1,3), feature_vectors, ls, iter_nr)
        rgb = rgb_flat.reshape(-1, nr_samples_per_ray, 3)
        # print("rgb is ", rgb.shape)
        # print("samples_depth", samples_depth.shape)

        weights = volume_rendering_volsdf(samples_depth.view(nr_rays, nr_samples_per_ray), sdf)
        # print("weights",weights.shape)
        # print("weight is ", weights)

        mask_inside_bb=check_point_inside_cube(ray_samples, self.bounding_box_sizes_xyz, self.bounding_box_translation)*1.0
        # rgb_sigma=rgb_sigma*mask_inside_bb

        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb *mask_inside_bb, 1)

        # print("rgb_values", rgb_values.shape)

        
        
       


        return rgb_values 

    # def volume_rendering(self, z_vals, sdf):
    #     density_flat = self.density(sdf)
    #     density = density_flat.reshape(-1, z_vals.shape[1])  # (batch_size * num_pixels) x N_samples

    #     dists = z_vals[:, 1:] - z_vals[:, :-1]
    #     dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)

    #     # print("dists", dists.shape)

    #     # LOG SPACE
    #     free_energy = dists * density
    #     shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)  # shift one step
    #     alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
    #     transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
    #     weights = alpha * transmittance # probability of the ray hits something here

    #     return weights


#########SDF######################################33
class SirenMLP(torch.nn.Module):

    def __init__(self, in_channels, hidden_dim, out_channels, nr_layers, scale_init):
        super(SirenMLP, self).__init__()

        self.layers=[]
        self.layers.append(  BlockSiren2(in_channels=in_channels, out_channels=hidden_dim, is_first_layer=True, scale_init=scale_init ) )
        for i in range(nr_layers):
            self.layers.append(   BlockSiren2(in_channels=hidden_dim, out_channels=hidden_dim )  )
        self.layers.append(  BlockSiren2(in_channels=hidden_dim, out_channels=out_channels, activ=None )  )

        self.mlp=torch.nn.Sequential(*self.layers)

        self.weights_initialized=True #so that apply_weight_init_fn doesnt initialize anything

    
    def forward(self, x):

        x=self.mlp(x)

        return x


class GaussMLP(torch.nn.Module):

    def __init__(self, in_channels, hidden_dim, out_channels, nr_layers, sigma):
        super(GaussMLP, self).__init__()

        self.layers=[]
        self.layers.append(  
            nn.Sequential(
                LinearWN(in_channels, hidden_dim),
                GaussActiv(sigma)
            )
        )
        for i in range(nr_layers):
            self.layers.append(   
                nn.Sequential(
                    LinearWN(hidden_dim, hidden_dim),
                    GaussActiv(sigma)
                )
            )
        self.layers.append(  
            nn.Sequential(
                LinearWN(hidden_dim, out_channels),
            )
        )

        self.mlp=torch.nn.Sequential(*self.layers)

        apply_weight_init_fn(self, leaky_relu_init, negative_slope=0.0)
        leaky_relu_init(self.mlp[-1], 1.0)

        self.weights_initialized=True #so that apply_weight_init_fn doesnt initialize anything

    
    def forward(self, x):

        x=self.mlp(x)

        return x


#given positions and features, decodes something
class ModulatedSiren(torch.nn.Module):
    #a siren network which predicts various direction vectors along the strand similar ot FakeODE. the idea is that siren works well when periodic thing needs to be predicted and the strand can be seen as some periodic direction vectors being repeted at some points on the strand
    #the idea is similar to modulation siren https://arxiv.org/pdf/2104.03960.pdf
    def __init__(self, in_channels_positions, in_channels_features, hidden_dim, out_channels, nr_layers, scale_init, flip_modulation):
        super(ModulatedSiren, self).__init__()

        self.flip_modulation=flip_modulation

        # self.swish=th.nn.SiLU()
        self.activ=torch.nn.Mish()
        self.tanh=th.nn.Tanh()

        self.nr_layers=nr_layers

        cur_nr_channels=in_channels_features
        self.modulation_layers=torch.nn.ModuleList([])
        for i in range(nr_layers):
            self.modulation_layers.append( LinearWN(cur_nr_channels, hidden_dim) )
            cur_nr_channels= hidden_dim+in_channels_features  #at the end we concatenate the input z
        self.last=LinearWN(hidden_dim, out_channels)

        apply_weight_init_fn(self, leaky_relu_init, negative_slope=0.0)
        leaky_relu_init(self.last, 1.0)





        #siren layers which are put here because we don't want them to be intiialized by swift
        self.siren_layers=torch.nn.ModuleList([])
        self.siren_layers.append(  BlockSiren2(in_channels=in_channels_positions, out_channels=hidden_dim, is_first_layer=True, scale_init=scale_init)   )
        for i in range(self.nr_layers-1):
            self.siren_layers.append(  BlockSiren2(in_channels=hidden_dim, out_channels=hidden_dim)   )



    def forward(self, positions, features ):


        h_siren=positions
        # z_scaling=0.001 #this has to be initialized so that the h_modulation is something like 0.2.If its lower, then no gradient will flow into Z and then the network will not be optimized. You might need to do one run and check the gradients of the network with model.summary to see if the gradients don't vanish
        z_scaling=1.0
        z=features
        z_initial=z*z_scaling
        z=z*z_scaling
        with_checkpointing=True
        for i in range(self.nr_layers):
            h_modulation= self.activ( self.modulation_layers[i](z) )
            # print("h_modulation mean and std", h_modulation.mean(), " ", h_modulation.std())
            # print("h_modulation min max", h_modulation.min(), " ", h_modulation.max())
            s=self.siren_layers[i](h_siren)
            if self.flip_modulation:
                h_siren=(1-h_modulation)*s
            else:
                h_siren=h_modulation*s
            # h_siren=s
            
            #for next iter
            z=torch.cat([z_initial, h_modulation],1)

        #last
        x=self.last(h_siren)
         
        return x

class ModulatedGauss(torch.nn.Module):
    #a siren network which predicts various direction vectors along the strand similar ot FakeODE. the idea is that siren works well when periodic thing needs to be predicted and the strand can be seen as some periodic direction vectors being repeted at some points on the strand
    #the idea is similar to modulation siren https://arxiv.org/pdf/2104.03960.pdf
    def __init__(self, in_channels_positions, in_channels_features, hidden_dim, out_channels, nr_layers, sigma, flip_modulation):
        super(ModulatedGauss, self).__init__()

        self.flip_modulation=flip_modulation

        # self.swish=th.nn.SiLU()
        self.activ=torch.nn.Mish()
        self.tanh=th.nn.Tanh()

        self.nr_layers=nr_layers

        cur_nr_channels=in_channels_features
        self.modulation_layers=torch.nn.ModuleList([])
        for i in range(nr_layers):
            self.modulation_layers.append( LinearWN(cur_nr_channels, hidden_dim) )
            cur_nr_channels= hidden_dim+in_channels_features  #at the end we concatenate the input z
        self.last=LinearWN(hidden_dim, out_channels)

        apply_weight_init_fn(self, leaky_relu_init, negative_slope=0.0)
        leaky_relu_init(self.last, 1.0)




        # nn.Sequential(
        #     LinearWN(nr_lattice_features*nr_resolutions  ,32),
        #     GaussActiv(sigma)
        # )


        #siren layers which are put here because we don't want them to be intiialized by swift
        self.siren_layers=torch.nn.ModuleList([])
        self.siren_layers.append(  
            # BlockSiren(in_channels=in_channels_positions, out_channels=hidden_dim, is_first_layer=True, scale_init=scale_init)   
            nn.Sequential(
                LinearWN(in_channels_positions, hidden_dim),
                GaussActiv(sigma)
            )
        )
        for i in range(self.nr_layers-1):
            self.siren_layers.append(  
                # BlockSiren(in_channels=hidden_dim, out_channels=hidden_dim)   
                nn.Sequential(
                    LinearWN(hidden_dim, hidden_dim),
                    GaussActiv(sigma)
                )
            )



    def forward(self, positions, features ):


        h_siren=positions
        # z_scaling=0.001 #this has to be initialized so that the h_modulation is something like 0.2.If its lower, then no gradient will flow into Z and then the network will not be optimized. You might need to do one run and check the gradients of the network with model.summary to see if the gradients don't vanish
        z_scaling=1.0
        z=features
        z_initial=z*z_scaling
        z=z*z_scaling
        with_checkpointing=True
        for i in range(self.nr_layers):
            h_modulation= self.activ( self.modulation_layers[i](z) )
            # print("h_modulation mean and std", h_modulation.mean(), " ", h_modulation.std())
            # print("h_modulation min max", h_modulation.min(), " ", h_modulation.max())
            s=self.siren_layers[i](h_siren)
            if self.flip_modulation:
                h_siren=(1-h_modulation)*s
            else:
                h_siren=h_modulation*s
            
            #for next iter
            z=torch.cat([z_initial, h_modulation],1)

        #last
        x=self.last(h_siren)
         
        return x

class SDFnoMLP(torch.nn.Module):

    def __init__(self, nr_lattice_vertices, nr_lattice_features, nr_resolutions, boundary_primitive, feat_size_out, N_initial_samples, N_samples_importance, N_iters_upsample):
        super(SDFnoMLP, self).__init__()

        self.boundary_primitive=boundary_primitive

        self.nr_lattice_features=nr_lattice_features

        self.reduction_lattice="concat" #sum, concat
        # self.reduction_lattice="sum" #sum, concat

        # self.lattice_type="grid" #grid, permuto
        # self.lattice_type="permuto" #grid, permuto
        self.lattice_type="permuto_monolithic" #the same as permuto but it does the slicing from all levels at the same time and it's way faster
        # self.lattice_type="hashgrid" #hashgrid
        # self.lattice_type="hashgrid_fused"

        
        self.feat_size_out=feat_size_out


        self.pick_rand_rows= RandRowPicker()
        self.pick_rand_pixels= RandPixelPicker(low_discrepancy=False)
        # self.pick_patch_and_rand_pixels= PatchAndRandPixelPicker()
        self.row_sampler=RowSampler()
        self.create_rays=CreateRaysModule()
        #original values from volsdf
        # N_samples=64
        # N_samples_eval=128 
        # N_samples_extra=32
        #less samples for faster speed
        # N_samples=32
        # N_samples_eval=64
        # N_samples_extra=0


        self.N_initial_samples=N_initial_samples
        self.N_samples_importance=N_samples_importance
        self.N_iters_upsample=N_iters_upsample
        
        # self.ray_sampler=UniformSampler(boundary_primitive=boundary_primitive, N_samples=60 ) #works
        # self.ray_sampler=ErrorBoundSampler(boundary_primitive=boundary_primitive, N_samples=64, N_samples_eval=128, N_samples_extra=32, eps=0.1, beta_iters=10, max_total_iters=5, add_tiny=1e-6)
        # self.volume_renderer=VolumeRenderingVolSDF()





        self.nr_resolutions=nr_resolutions
        self.slice_lattice=SliceLatticeWithCollisionsModule()
        # self.slice_lattice=SliceLatticeWithCollisionsDifferentiableModule(smooth_barycentric=False)
        self.slice_lattice_monolithic=SliceLatticeWithCollisionsFastMRMonolithicModule()


        self.num_encoding_functions_pos=0
        self.pos_encode_pos=PositionalEncoding(in_channels=3, num_encoding_functions=self.num_encoding_functions_pos, only_sin=False) 
        # self.pos_encode_pos=PositionalEncodingRandFeatures(in_channels=3, num_encoding_functions=self.num_encoding_functions_pos, sigma=0.1) 
        self.nr_channels_after_encoding_pos=3+3*2*self.num_encoding_functions_pos
        # self.nr_channels_after_encoding_pos=0


        coarsest_sigma=1.0
        finest_sigma=0.0001
        # finest_sigma=0.0005
        self.sigmas_list=np.geomspace(coarsest_sigma, finest_sigma, num=nr_resolutions) #the smaller the value, the finer the lattice
        if self.lattice_type=="permuto":
            self.lattice_values_list = torch.nn.ParameterList()
            for i in range(nr_resolutions):
                self.lattice_values_list.append(  torch.nn.Parameter( torch.randn( nr_lattice_vertices, nr_lattice_features  )*1e-4 , requires_grad=True)  )

        if self.lattice_type=="permuto_monolithic":
            lattice_values_list_raw=[]
            init_list=np.geomspace(1e-4, 1e-6, num=nr_resolutions)
            for i in range(nr_resolutions):
                lattice_values_list_raw.append( torch.randn( nr_lattice_vertices, nr_lattice_features  )*1e-7  )
                #init with the initlist
                # init_val=init_list[i]
                # if i<nr_resolutions/2:
                    # init_val=1e-4
                # print("init val is ", init_val)
                # lattice_values_list_raw.append( torch.randn( nr_lattice_vertices, nr_lattice_features  )*init_val  )
            #make a monolithic version because we want to also use the monolithic slicer
            self.lattice_values_monolithic=torch.cat(lattice_values_list_raw,1)
            self.lattice_values_monolithic=self.lattice_values_monolithic.view(nr_lattice_vertices, nr_resolutions, nr_lattice_features).permute(1,0,2).contiguous() #get it to [nr_resolutions, nr_vertices, nr_features]
            self.lattice_values_monolithic=torch.nn.Parameter(self.lattice_values_monolithic)

        elif self.lattice_type=="grid":
            #make dense 3D grid
            coarsest_grid_size=2
            finest_grid_size=128
            self.grid_size_list=np.geomspace(coarsest_grid_size, finest_grid_size, num=nr_resolutions)
            self.grid_3d_list = torch.nn.ParameterList()
            for i in range(nr_resolutions):
                cur_grid_size=int(self.grid_size_list[i])
                print("cur_grid_size",cur_grid_size)
                self.grid_3d_list.append( torch.nn.Parameter( torch.randn( 1, nr_lattice_features, cur_grid_size, cur_grid_size, cur_grid_size  )*1e-4 , requires_grad=True)  )
        elif self.lattice_type=="hashgrid":
            config_encoding={
                "otype": "HashGrid",
                "n_levels": nr_resolutions,
                "n_features_per_level": nr_lattice_features,
                "log2_hashmap_size": 18,
                "base_resolution": 16
                # "interpolation": "Linear"
            }
            self.encoding = tcnn.Encoding(n_input_dims=3, encoding_config=config_encoding)
        elif self.lattice_type=="hashgrid_fused":
            config_encoding={
                "otype": "HashGrid",
                "n_levels": nr_resolutions,
                "n_features_per_level": nr_lattice_features,
                "log2_hashmap_size": 19,
                "base_resolution": 16
            }
            mlp_tcnn_config={
                "otype": "FullyFusedMLP",
                "activation": "Softplus",
                "output_activation": "None",
                "n_neurons": 32,
                "n_hidden_layers": 2
            }
            self.fused_encode_mlp = tcnn.NetworkWithInputEncoding(
                3, 1,
                config_encoding, mlp_tcnn_config
            )
            self.sdf_shift=0.0



        nr_resolutions_to_slice=nr_resolutions
        if self.reduction_lattice=="sum":
            nr_resolutions_to_slice=1

        #randomly shift the cloud at each resolution in order to avoid too many hash collisions
        self.random_shift_list=[]
        for i in range(nr_resolutions):
            self.random_shift_list.append(   torch.randn( 1, 3)*10  )
        self.random_shift_monolithic=torch.nn.Parameter( torch.cat(self.random_shift_list,0) ) #we make it a parameter just so it gets saved when we checkpoint
            

        # self.sdf_shift=-0.5
        # self.sdf_shift=1.0
        # self.sdf_shift=3.6
        # self.sdf_shift=0.0
        # self.sdf_shift=0.1
        # self.sdf_shift=1e-5
        # self.sdf_shift=1e-3 #the lowest it can go and still converges
        self.sdf_shift=1e-2
        # self.sdf_shift=1e-1
        # self.sdf_shift=0.2
        # self.sdf_shift=0.15
        self.mlp_sdf= nn.Sequential(
            # LinearWN(nr_lattice_features*nr_resolutions_to_slice,32),
            # LinearWN(nr_lattice_features*nr_resolutions_to_slice+3,32),
            torch.nn.Linear(nr_lattice_features*nr_resolutions_to_slice ,nr_lattice_features*nr_resolutions_to_slice),
            # torch.nn.Mish(),
            # torch.nn.Linear(32,32),
            # torch.nn.Mish(),
            # torch.nn.Linear(32,32),
            torch.nn.Mish(),
            torch.nn.Linear(nr_lattice_features*nr_resolutions_to_slice, nr_lattice_features*nr_resolutions_to_slice)
        )
        apply_weight_init_fn(self, leaky_relu_init, negative_slope=0.0)
        leaky_relu_init(self.mlp_sdf[-1], negative_slope=1.0)
        # r_init=10
        # self.mlp_sdf[0].fuse()
        # self.mlp_sdf[0].weight.data.uniform_(-r_init, r_init)
        # self.mlp_sdf[0].unfuse()
        #init first linear because the input channels corresponding to the lattice features are actually mostly zero
        # gain = np.sqrt(2.0 / (1.0 + 0.0 ** 2))
        # self.mlp_sdf[0].fuse()
        # n1 = 0.1
        # n2 = self.mlp_sdf[0].out_features
        # std = gain * np.sqrt(2.0 / (n1 + n2))
        # self.mlp_sdf[0].weight.data.uniform_(-std * np.sqrt(3.0), std * np.sqrt(3.0))
        # self.mlp_sdf[0].unfuse()



        #with tinycudnn
        # self.sdf_shift=-0.5
        # mlp_tcnn_config={
        #     "otype": "FullyFusedMLP",
        #     "activation": "Softplus",
        #     "output_activation": "None",
        #     "n_neurons": 32,
        #     "n_hidden_layers": 2
        # }
        # self.mlp_sdf = tcnn.Network(
        #                 nr_lattice_features*nr_resolutions_to_slice, 
        #                 1+feat_size_out, 
        #                 mlp_tcnn_config) 


        # self.sdf_shift=1.5
        # self.mlp_sdf= nn.Sequential(
        #     torch.nn.Linear(nr_lattice_features*nr_resolutions_to_slice,32),
        #     torch.nn.Softplus(),
        #     torch.nn.Linear(32,32),
        #     torch.nn.Softplus(),
        #     torch.nn.Linear(32,32),
        #     torch.nn.Softplus(),
        #     torch.nn.Linear(32,1)
        # )
        # apply_weight_init_fn(self, leaky_relu_init, negative_slope=0.0)
        # leaky_relu_init(self.mlp_sdf[-1], negative_slope=1.0)


        # self.sdf_shift=0.5
        # self.mlp_sdf= SirenMLP(in_channels=nr_lattice_features*nr_resolutions_to_slice+ self.nr_channels_after_encoding_pos, hidden_dim=32, out_channels=1+feat_size_out, nr_layers=2, scale_init=2)
        # self.mlp_sdf= SirenMLP(in_channels=3, hidden_dim=128, out_channels=1+feat_size_out, nr_layers=6, scale_init=30)

        # self.sdf_shift=0.7
        # self.mlp_sdf= GaussMLP(in_channels=nr_lattice_features*nr_resolutions_to_slice, hidden_dim=32, out_channels=1+feat_size_out, nr_layers=2, sigma=0.1)



        #modulated siren
        # self.sdf_shift=1e-3
        # self.mod_sin=ModulatedSiren(in_channels_positions=3, in_channels_features=nr_lattice_features*nr_resolutions_to_slice + self.nr_channels_after_encoding_pos, hidden_dim=32, out_channels=1+feat_size_out, nr_layers=3, scale_init=30, flip_modulation=False)
        # self.mod_gauss=ModulatedGauss(in_channels_positions=3, in_channels_features=nr_lattice_features*nr_resolutions_to_slice, hidden_dim=32, out_channels=1, nr_layers=3, sigma=0.1, flip_modulation=True)




        # nr_hidden=64
        # start_nr_channels=3+nr_lattice_features*nr_resolutions
        # self.mlp_sdf_list=torch.nn.ModuleList([])
        # self.mlp_sdf_list.append(   torch.nn.Linear(start_nr_channels, nr_hidden )   )
        # for i in range(2):
        #     self.mlp_sdf_list.append(   torch.nn.Linear(nr_hidden, nr_hidden)   )
        # self.mlp_sdf_list.append(   torch.nn.Linear(nr_hidden, 1)   )

        # #geometric init https://arxiv.org/pdf/1911.10414.pdf
        # p=1.0
        # lin=self.mlp_sdf_list[0]
        # torch.nn.init.normal_(lin.weight, mean=2*np.sqrt(np.pi) / np.sqrt(p * start_nr_channels), std=0.000001)
        # torch.nn.init.constant_(lin.bias, -1.0)
        # for i in range(len(self.mlp_sdf_list)-2):
        #     lin=self.mlp_sdf_list[i+1]
        #     torch.nn.init.normal_(lin.weight, mean=2*np.sqrt(np.pi) / np.sqrt(p * nr_hidden), std=0.000001)
        #     torch.nn.init.constant_(lin.bias, -1.0)
        # lin=self.mlp_sdf_list[-1]
        # torch.nn.init.constant_(lin.bias, 0.0)
        # torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(p*1))

        self.ray_sampler=NeusSampler(boundary_primitive=boundary_primitive, N_initial_samples=self.N_initial_samples, N_samples_importance=self.N_samples_importance, n_iters_upsample=self.N_iters_upsample)
        self.volume_renderer=VolumeRenderingNeus(nr_lattice_vertices,  ingle_variance=True)
        # self.volume_renderer=VolumeRenderingVolSDF(set_last_weight_zero=True, normalize_weights=False)
        self.softplus=torch.nn.Softplus()



        self.c2f=Coarse2Fine(self.nr_resolutions)
        self.hook=None
        self.hook_layer=None
        # self.nr_iters_for_c2f=5000
        self.nr_iters_for_c2f=10000
    
    def forward(self, points, ls, iter_nr ):

        # print("iter_nr model", iter_nr)


        assert points.shape[1] == 3, "ray_samples_flat should be nx3"

        # window=self.c2f(iter_nr*10.0001+0.3)
        # window=self.c2f(iter_nr*0.0003+0.3)
        # window=self.c2f(iter_nr*0.0001+0.3)
        # window=self.c2f( map_range_val(iter_nr, 0.0, 25000, 0.3, 1.0   ) )
        window=self.c2f( map_range_val(iter_nr, 0.0, self.nr_iters_for_c2f, 0.3, 1.0   ) )
        # window=self.c2f(iter_nr*0.001+0.3)
        # window=self.c2f(iter_nr*0.00003+0.3)
        # window=(window>0.0)*1.0 #we do hard steps so that the network doesnt start learning very high values for the windowed dimensions that start annealing with a low value
        # print("window", window)

        if self.lattice_type=="hashgrid_fused":
            sdf=self.fused_encode_mlp(points)
            feat=None
            sdf=sdf.float()
            return sdf, feat

        ray_features_full=None
        TIME_START("slice_full_sdf")
        if self.lattice_type=="permuto":
            ray_features_list=[]
            for i in range(self.nr_resolutions):
                ls.set_sigma(self.sigmas_list[i])
                values=self.lattice_values_list[i] * window[i]
                ray_features, splatting_indices, splatting_weights=self.slice_lattice(values, ls, points )
                ray_features_list.append(ray_features)
        elif self.lattice_type=="permuto_monolithic":
            lattice_values_monolithic_annealed=self.lattice_values_monolithic * window.view(-1,1,1)
            ray_features_full, splatting_indices, splatting_weights=self.slice_lattice_monolithic(lattice_values_monolithic_annealed, self.sigmas_list, ls, points, self.random_shift_monolithic.detach() )
            #anneal the gradient so that we get a coarse to fine update
            # if self.hook is None and self.training:
            #     print("---------registering hook")
            #     self.hook = self.lattice_values_monolithic.register_hook(lambda grad: grad* window.view(-1,1,1)  )  # double the gradient
            #     window_layer=window.view(-1,1).repeat(1,self.nr_lattice_features).view(-1,1) #nr_res * nr_feat x 1
            #     window_layer=window_layer.permute(1,0) #1 x nrres*nrfeat
            #     window_layer=torch.cat( [window_layer, torch.ones(1,3)], -1 ) #the rest of the featues like point positions
            #     self.hook_layer = self.mlp_sdf[0].weight.register_hook(lambda grad: grad*window_layer  )

            #concat also the splatting weight for each level
            #splatting indices and weights are of shape nr_resolutions, (pos_dim+1), nr_positions 
            # splatting_weights_11=splatting_weights*2-1
            # splatting_weights_11=splatting_weights_11.permute(2,0,1)#go from lvl, 4, nr_positions to nr_positions,lvl,4
            # splatting_weights_11=splatting_weights_11.view(points.shape[0],self.nr_resolutions*4)
            # ray_features_full=torch.cat([ray_features_full,splatting_weights_11*1e-3],1)
            #concat also the points themselves
            # ray_features_full=torch.cat([ray_features_full,points*1e-3],1)

            #check the nr of points vs nr of unique lattice vertices we access at the fines lvl
            # print("pts is", points.shape)
            # print("splatting_indices",splatting_indices.shape)
            # if splatting_indices.shape[0]==24:
                # splatting_indices_finest=splatting_indices[23,:,:]
                # print("splatting_indices_finest", splatting_indices_finest.shape)
                # unique_indices=torch.unique(splatting_indices_finest)
                # print("unique_indices", unique_indices.shape)
                # exit(1)

        elif self.lattice_type=="grid":
            #slice 3d grid
            points_scaled=points*1.9 #scale from [-0.5, 0.5] to [-1, 1]
            points_scaled=points_scaled.view(1,1,1,-1,3)
            ray_features_list=[]
            for i in range(len(self.grid_3d_list)):
                values= self.grid_3d_list[i] * window[i]
                # ray_features=grid_sample_3d(values, points_scaled)
                ray_features=torch.nn.functional.grid_sample(values, points_scaled, mode='bilinear')
                ray_features=ray_features.transpose(1,4)
                ray_features=ray_features.view(-1,self.nr_lattice_features)
                ray_features_list.append(ray_features)
        elif self.lattice_type=="hashgrid":
            ray_features_full=self.encoding(points)
            ray_features_full=ray_features_full.float()
            # print("ray_features_full, ,in max ", ray_features_full.min(), ray_features_full.max())
            # print("ray_features_full", ray_features_full.shape)
        TIME_END("slice_full_sdf")


        # #reduce the features
        if  ray_features_full == None: #if we don't have the full features it means we have to reduce the list.
            if self.reduction_lattice=="concat":
                ray_features_full=torch.cat(ray_features_list, 1)
            elif self.reduction_lattice=="sum":
                ray_features_full=torch.stack(ray_features_list, dim=0).sum(dim=0)
                print("ray_features_full", ray_features_full.shape)


            
        # ray_features_full=torch.cat([ray_features_full,points],1)
        # ray_features_full=torch.sin(ray_features_full)
        # sdf, _=ray_features_full.min(dim=1, keepdim=True)
        # ray_features_full=self.mlp_sdf(ray_features_full)
        sdf=ray_features_full.mean(dim=1, keepdim=True) #kinda noisy but works
        # sdf=torch.sin(sdf)
        # sdf=ray_features_full.logsumexp(dim=1, keepdim=True) 
        # print("sdf", sdf.shape)
        # sdf=ray_features_full.prod(dim=1, keepdim=True) 
        sdf=sdf+self.sdf_shift
        feat=None
        # print("ray_features_full", ray_features_full.shape)


        # #concat also frequency encoded points which help to resolve collisions
        # # points_encoded=self.pos_encode_pos(points)
        # points_encoded=points
        # ray_features_full=torch.cat([ray_features_full,points_encoded*1e-3],1)


        # x=torch.cat([ray_features_full],1)
        # TIME_START("SDF_mlp")
        # sdf_and_feat=self.mlp_sdf(x)
        # # sdf_and_feat=self.mlp_sdf(points)
        # # sdf_and_feat=self.mod_sin(points, ray_features_full)
        # TIME_END("SDF_mlp")
        # # sdf=sdf_and_feat[:,0:1]
        # sdf=sdf_and_feat[:,0:1]
        # # sdf=0.2+sdf*0.1
        # # print("sdf min max ", sdf.min(), sdf.max())
        # sdf=sdf+self.sdf_shift
        # if self.feat_size_out!=0:
        #     feat=sdf_and_feat[:,-self.feat_size_out:]
        # else:
        #     feat=None


        



        return sdf, feat

    def get_sdf_and_gradient(self, points, ls, iter_nr):

        method="finite_difference" #autograd, finite_difference
        # method="autograd" #autograd, finite_difference
        # do_comparison=True #when running finite difference, run also autograd and check that they give rougly the same result
        do_comparison=False #when running finite difference, run also autograd and check that they give rougly the same result

        #dummy
        # sdf, feat = self.forward(points, ls, iter_nr)
        # return sdf, points.clone(), feat

        if method=="finite_difference":
            # do it by finite differences because doing it with autograd doesnt work with the lattice since it would require a double backward through the splatting and slicing function

            #to the original positions, add also a tiny epsilon in all directions
            nr_points_original=points.shape[0]
            epsilon=1e-4
            points_xplus=points.clone()
            points_yplus=points.clone()
            points_zplus=points.clone()
            points_xplus[:,0]=points_xplus[:,0]+epsilon
            points_yplus[:,1]=points_yplus[:,1]+epsilon
            points_zplus[:,2]=points_zplus[:,2]+epsilon

            points_full=torch.cat([points, points_xplus, points_yplus, points_zplus],0)

            sdf_full, feat_full = self.forward(points_full, ls, iter_nr)

            feat=None
            if feat_full is not None:            
                feats=feat_full.chunk(4, dim=0) 
                feat=feats[0]

            sdfs=sdf_full.chunk(4, dim=0) 
            sdf=sdfs[0]
            sdf_xplus=sdfs[1]
            sdf_yplus=sdfs[2]
            sdf_zplus=sdfs[3]


            grad_x=(sdf_xplus-sdf)/epsilon
            grad_y=(sdf_yplus-sdf)/epsilon
            grad_z=(sdf_zplus-sdf)/epsilon
            
            gradients=torch.cat([grad_x, grad_y, grad_z],1)


            if do_comparison:
                #do it with autograd
                with torch.set_grad_enabled(True):
                    points.requires_grad_(True)
                    sdf, feat = self.forward(points, ls, iter_nr)
                    # ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
                    # if self.sdf_bounding_sphere > 0.0:
                    #     sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
                    #     sdf = torch.minimum(sdf, sphere_sdf)
                    # feature_vectors = output[:, 1:]

                    feature_vectors=None
                    d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
                    gradients_autograd = torch.autograd.grad(
                        outputs=sdf,
                        inputs=points,
                        grad_outputs=d_output,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True)[0]
                diff=((gradients-gradients_autograd)**2).mean()
                if diff.item()>1e-6:
                    print("error is to high. diff is ", diff)
                    print("gradients finite_diff is ", gradients)
                    print("gradients autograd is ", gradients_autograd)
                    exit(1)



        elif method=="autograd":

            #do it with autograd
            with torch.set_grad_enabled(True):
                points.requires_grad_(True)
                sdf, feat = self.forward(points, ls, iter_nr)
                # ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
                # if self.sdf_bounding_sphere > 0.0:
                #     sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
                #     sdf = torch.minimum(sdf, sphere_sdf)
                # feature_vectors = output[:, 1:]

                feature_vectors=None
                d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
                gradients = torch.autograd.grad(
                    outputs=sdf,
                    inputs=points,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)[0]
        else:
            print("method not known")
            exit(1)




        return sdf, gradients, feat

    def get_sdf_and_curvature_1d_precomputed_gradient(self, points, sdf_gradients, ls, iter_nr):
        #get the curvature along a certain random direction for each point

        method="finite_difference" #autograd, finite_difference


        if method=="finite_difference":
            # do it by finite differences because doing it with autograd doesnt work with the lattice since it would require a double backward through the splatting and slicing function
            #do it by taking the central point and then the p+epsilon and p-epsilon in some random directions

            #to the original positions, add also a tiny epsilon 
            nr_points_original=points.shape[0]
            epsilon=1e-4
            # epsilon_end=1e-4
            # epsilon=map_range_val(iter_nr, 0, 10000, 1e-2, epsilon_end)
            # epsilon=np.random.uniform(1e-4,1e-3)
            rand_directions=torch.randn_like(points)
            rand_directions=F.normalize(rand_directions,dim=-1)

            #instead of random direction we take the normals at these points, and calculate a random vector that is orthogonal 
            # with torch.set_grad_enabled(True):
            #     sdf, sdf_gradients, feat=self.get_sdf_and_gradient(points, ls, iter_nr)
            #     normals=F.normalize(sdf_gradients,dim=-1)
            #     tangent=torch.cross(normals, rand_directions)
            #     rand_directions=tangent #set the random moving direction to be the tangent direction now

            normals=F.normalize(sdf_gradients,dim=-1)
            # normals=normals.detach()
            tangent=torch.cross(normals, rand_directions)
            rand_directions=tangent #set the random moving direction to be the tangent direction now

            

            points_plus=points.clone()+rand_directions*epsilon
            points_minus=points.clone()-rand_directions*epsilon

            points_full=torch.cat([points, points_plus, points_minus],0)

            sdf_full, feat_full = self.forward(points_full, ls, iter_nr)

            feat=None
            if feat_full is not None:            
                feats=feat_full.chunk(3, dim=0) 
                feat=feats[0]

            sdfs=sdf_full.chunk(3, dim=0) 
            sdf=sdfs[0]
            sdf_plus=sdfs[1]
            sdf_minus=sdfs[2]

            curvature=(sdf_plus-2*sdf+sdf_minus)/(epsilon*epsilon)
            


           

        else:
            print("method not known")
            exit(1)




        return sdf, curvature, feat

    def get_sdf_and_curvature_1d(self, points, ls, iter_nr):
        #get the curvature along a certain random direction for each point

        method="finite_difference" #autograd, finite_difference


        if method=="finite_difference":
            # do it by finite differences because doing it with autograd doesnt work with the lattice since it would require a double backward through the splatting and slicing function
            #do it by taking the central point and then the p+epsilon and p-epsilon in some random directions

            #to the original positions, add also a tiny epsilon 
            nr_points_original=points.shape[0]
            epsilon=1e-4
            rand_directions=torch.randn_like(points)
            rand_directions=F.normalize(rand_directions,dim=-1)

            # instead of random direction we take the normals at these points, and calculate a random vector that is orthogonal 
            with torch.set_grad_enabled(True):
                sdf, sdf_gradients, feat=self.get_sdf_and_gradient(points, ls, iter_nr)
                normals=F.normalize(sdf_gradients,dim=-1)
                tangent=torch.cross(normals, rand_directions)
                rand_directions=tangent #set the random moving direction to be the tangent direction now

            normals=F.normalize(sdf_gradients,dim=-1)
            tangent=torch.cross(normals, rand_directions)
            rand_directions=tangent #set the random moving direction to be the tangent direction now

            

            points_plus=points.clone()+rand_directions*epsilon
            points_minus=points.clone()-rand_directions*epsilon

            points_full=torch.cat([points, points_plus, points_minus],0)

            sdf_full, feat_full = self.forward(points_full, ls, iter_nr)

            feat=None
            if feat_full is not None:            
                feats=feat_full.chunk(3, dim=0) 
                feat=feats[0]

            sdfs=sdf_full.chunk(3, dim=0) 
            sdf=sdfs[0]
            sdf_plus=sdfs[1]
            sdf_minus=sdfs[2]

            curvature=(sdf_plus-2*sdf+sdf_minus)/(epsilon*epsilon)
            


           

        else:
            print("method not known")
            exit(1)




        return sdf, curvature, feat

    def get_sdf_and_curvature_1d_precomputed_gradient_normal_based(self, points, sdf_gradients, ls, iter_nr):
        #get the curvature along a certain random direction for each point
        #does it by computing the normal at a shifted point on the tangent plant and then computing a dot produt

        method="finite_difference" #autograd, finite_difference


        if method=="finite_difference":
            # do it by finite differences because doing it with autograd doesnt work with the lattice since it would require a double backward through the splatting and slicing function
            #do it by taking the central point and then the p+epsilon and p-epsilon in some random directions

            #to the original positions, add also a tiny epsilon 
            nr_points_original=points.shape[0]
            epsilon=1e-4
            # epsilon=np.random.uniform(1e-4,1e-3)
            rand_directions=torch.randn_like(points)
            rand_directions=F.normalize(rand_directions,dim=-1)

            #instead of random direction we take the normals at these points, and calculate a random vector that is orthogonal 
            # with torch.set_grad_enabled(True):
            #     sdf, sdf_gradients, feat=self.get_sdf_and_gradient(points, ls, iter_nr)
            #     normals=F.normalize(sdf_gradients,dim=-1)
            #     tangent=torch.cross(normals, rand_directions)
            #     rand_directions=tangent #set the random moving direction to be the tangent direction now

            normals=F.normalize(sdf_gradients,dim=-1)
            # normals=normals.detach()
            tangent=torch.cross(normals, rand_directions)
            rand_directions=tangent #set the random moving direction to be the tangent direction now

            

            points_shifted=points.clone()+rand_directions*epsilon
            
            #get the gradient at the shifted point
            sdf_shifted, sdf_gradients_shifted, feat_shifted=self.get_sdf_and_gradient(points_shifted, ls, iter_nr) 

            normals_shifted=F.normalize(sdf_gradients_shifted,dim=-1)

            dot=(normals*normals_shifted).sum(dim=-1, keepdim=True)
            curvature=1-dot

            # curvature= ((normals-normals_shifted)**2 ).sum(dim=-1, keepdim=True)



           

        else:
            print("method not known")
            exit(1)




        return sdf_shifted, curvature, feat_shifted

    def save(self, root_folder, experiment_name, iter_nr):

        models_path=os.path.join(root_folder,"checkpoints/", experiment_name, str(iter_nr), "models")
        os.makedirs(models_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(models_path, "sdf_model.pt")  )

#uses a dense grid for the first dimension of 64x64x64 which is just the same nr of parameters as the lattice
#this dense grid stores directly sdf values and van be used directly to skip a lot of empty space when tracing 
#the rest of dimensions are done by a permutohedral lattice that just outputs a delta sdf towards the coarse value and a fature vector
class SDFDenseAndHash(torch.nn.Module):

    def __init__(self, nr_lattice_vertices, nr_lattice_features, nr_resolutions, boundary_primitive, feat_size_out, nr_iters_for_c2f, N_initial_samples, N_samples_importance, N_iters_upsample):
        super(SDFDenseAndHash, self).__init__()

        self.boundary_primitive=boundary_primitive

        self.nr_lattice_features=nr_lattice_features

        self.reduction_lattice="concat" #sum, concat
        # self.reduction_lattice="sum" #sum, concat

        # self.lattice_type="grid" #grid, permuto
        # self.lattice_type="permuto" #grid, permuto
        self.lattice_type="permuto_monolithic" #the same as permuto but it does the slicing from all levels at the same time and it's way faster
        # self.lattice_type="hashgrid" #hashgrid
        # self.lattice_type="hashgrid_fused"

        
        self.feat_size_out=feat_size_out


        self.pick_rand_rows= RandRowPicker()
        self.pick_rand_pixels= RandPixelPicker(low_discrepancy=False) #do NOT use los discrepancy for now, it seems to align some of the rays to some directions so maybe it's not that good of an idea
        # self.pick_patch_and_rand_pixels= PatchAndRandPixelPicker()
        self.row_sampler=RowSampler()
        self.create_rays=CreateRaysModule()
        #original values from volsdf
        # N_samples=64
        # N_samples_eval=128 
        # N_samples_extra=32
        #less samples for faster speed
        # N_samples=32
        # N_samples_eval=64
        # N_samples_extra=0


        self.N_initial_samples=N_initial_samples
        self.N_samples_importance=N_samples_importance
        self.N_iters_upsample=N_iters_upsample
        
        # self.ray_sampler=UniformSampler(boundary_primitive=boundary_primitive, N_samples=60 ) #works
        # self.ray_sampler=ErrorBoundSampler(boundary_primitive=boundary_primitive, N_samples=64, N_samples_eval=128, N_samples_extra=32, eps=0.1, beta_iters=10, max_total_iters=5, add_tiny=1e-6)
        # self.volume_renderer=VolumeRenderingVolSDF()





        self.nr_resolutions=nr_resolutions
        self.slice_lattice=SliceLatticeWithCollisionsModule()
        # self.slice_lattice=SliceLatticeWithCollisionsDifferentiableModule(smooth_barycentric=False)
        self.slice_lattice_monolithic=SliceLatticeWithCollisionsFastMRMonolithicModule()


        self.num_encoding_functions_pos=0
        self.pos_encode_pos=PositionalEncoding(in_channels=3, num_encoding_functions=self.num_encoding_functions_pos, only_sin=False) 
        # self.pos_encode_pos=PositionalEncodingRandFeatures(in_channels=3, num_encoding_functions=self.num_encoding_functions_pos, sigma=0.1) 
        self.nr_channels_after_encoding_pos=3+3*2*self.num_encoding_functions_pos
        # self.nr_channels_after_encoding_pos=0


        coarsest_sigma=1.0/64
        finest_sigma=0.0001
        # finest_sigma=0.0005
        self.sigmas_list=np.geomspace(coarsest_sigma, finest_sigma, num=nr_resolutions) #the smaller the value, the finer the lattice
        self.scale_factor=Lattice.compute_scale_factor_tensor(self.sigmas_list,3)


        #lattice stuff
        lattice_values_list_raw=[]
        init_list=np.geomspace(1e-4, 1e-6, num=nr_resolutions)
        for i in range(nr_resolutions):
            lattice_values_list_raw.append( torch.randn( nr_lattice_vertices, nr_lattice_features  )*1e-5  )
        #make a monolithic version because we want to also use the monolithic slicer
        self.lattice_values_monolithic=torch.cat(lattice_values_list_raw,1)
        self.lattice_values_monolithic=self.lattice_values_monolithic.view(nr_lattice_vertices, nr_resolutions, nr_lattice_features).permute(1,0,2).contiguous() #get it to [nr_resolutions, nr_vertices, nr_features]
        if Lattice.is_half_precision():
            print("switching to half precision lattice values")
            self.lattice_values_monolithic=self.lattice_values_monolithic.half()
        self.lattice_values_monolithic=torch.nn.Parameter(self.lattice_values_monolithic)


         

    
        #make dense 3D grid
        coarsest_grid_size=2
        finest_grid_size=128
        self.grid_size_list=np.geomspace(coarsest_grid_size, finest_grid_size, num=nr_resolutions)
        self.grid_3d_list = torch.nn.ParameterList()
        for i in range(len( self.grid_size_list)):
            cur_grid_size=int(self.grid_size_list[i])
            print("cur_grid_size",cur_grid_size)
            self.grid_3d_list.append( torch.nn.Parameter( torch.zeros( 1, 1, cur_grid_size, cur_grid_size, cur_grid_size  ) , requires_grad=True)  )

        # cur_grid_size=256
        # self.dense_grid_size=129
        self.dense_grid_size=64+1
        # self.dense_grid_size=32+1
        # self.dense_grid_size=4+1
        # self.dense_grid=torch.nn.Parameter( torch.randn( 1, 1, self.dense_grid_size, self.dense_grid_size, self.dense_grid_size  )*1e-4 , requires_grad=True)
        self.dense_grid=torch.nn.Parameter( torch.zeros( 1, 1, self.dense_grid_size, self.dense_grid_size, self.dense_grid_size  ) , requires_grad=True)
    
        #do it with my own voxelgrid
        self.dense_grid_custom=torch.nn.Parameter( torch.randn( self.dense_grid_size, self.dense_grid_size, self.dense_grid_size, 1  )*1e-5 , requires_grad=True)
        self.slice_voxel_grid=SliceVoxelGridModule(do_c2f=False, iterations_to_finish_c2f=1000)




        #randomly shift the cloud at each resolution in order to avoid too many hash collisions
        self.random_shift_list=[]
        for i in range(nr_resolutions):
            self.random_shift_list.append(   torch.randn( 1, 3)*10  )
        self.random_shift_monolithic=torch.nn.Parameter( torch.cat(self.random_shift_list,0) ) #we make it a parameter just so it gets saved when we checkpoint
            

        # self.sdf_shift=-0.5
        # self.sdf_shift=1.0
        # self.sdf_shift=3.6
        # self.sdf_shift=0.0
        # self.sdf_shift=0.1
        # self.sdf_shift=1e-5
        # self.sdf_shift=1e-3 #the lowest it can go and still converges
        self.sdf_shift=1e-2
        # self.sdf_shift=1e-1
        # self.sdf_shift=0.2
        # self.sdf_shift=0.15
        self.mlp_sdf= nn.Sequential(
            # LinearWN(nr_lattice_features*nr_resolutions_to_slice,32),
            # LinearWN(nr_lattice_features*nr_resolutions_to_slice+3,32),
            # torch.nn.Linear(nr_lattice_features*nr_resolutions_to_slice+ self.nr_channels_after_encoding_pos ,32),
            torch.nn.Linear(nr_lattice_features*nr_resolutions+ 4 ,32),
            # torch.nn.Linear(nr_lattice_features*nr_resolutions_to_slice ,32),
            # torch.nn.Linear(nr_lattice_features*nr_resolutions_to_slice+ self.nr_channels_after_encoding_pos ,64),
            # LinearWN(nr_lattice_features*nr_resolutions_to_slice + 4*nr_resolutions_to_slice,32),
            # torch.nn.Softplus(),
            torch.nn.Mish(),
            torch.nn.Linear(32,32),
            # torch.nn.Softplus(),
            torch.nn.Mish(),
            torch.nn.Linear(32,32),
            # torch.nn.Softplus(),
            torch.nn.Mish(),
            torch.nn.Linear(32,1+feat_size_out)
        )
        apply_weight_init_fn(self.mlp_sdf, leaky_relu_init, negative_slope=0.0)
        leaky_relu_init(self.mlp_sdf[-1], negative_slope=1.0)
        

        self.ray_sampler=NeusSampler(boundary_primitive=boundary_primitive, N_initial_samples=self.N_initial_samples, N_samples_importance=self.N_samples_importance, n_iters_upsample=self.N_iters_upsample)
        self.volume_renderer=VolumeRenderingNeus(nr_lattice_vertices, single_variance=True)
        # self.volume_renderer=VolumeRenderingVolSDF(set_last_weight_zero=True, normalize_weights=False)
        self.softplus=torch.nn.Softplus()

        # self.nr_iters_c2f_grid=1
        self.nr_iters_c2f_grid=1000
        # self.nr_iters_c2f_grid=300000
        self.c2f_grid_back_hook=Coarse2Fine3DGrid(4, self.nr_iters_c2f_grid)
        self.c2f=Coarse2Fine(self.nr_resolutions)
        self.c2f_grid=Coarse2Fine(self.nr_resolutions)
        self.hook=None
        self.hook_layer=None
        # self.nr_iters_for_c2f=5000
        # self.nr_iters_for_c2f=10000
        self.nr_iters_for_c2f=nr_iters_for_c2f



        # self.mlp_sdf_grid= nn.Sequential(
        #     # LinearWN(nr_lattice_features*nr_resolutions_to_slice,32),
        #     # LinearWN(nr_lattice_features*nr_resolutions_to_slice+3,32),
        #     # torch.nn.Linear(nr_lattice_features*nr_resolutions_to_slice+ self.nr_channels_after_encoding_pos ,32),
        #     torch.nn.Linear(nr_resolutions ,32),
        #     # torch.nn.Linear(nr_lattice_features*nr_resolutions_to_slice ,32),
        #     # torch.nn.Linear(nr_lattice_features*nr_resolutions_to_slice+ self.nr_channels_after_encoding_pos ,64),
        #     # LinearWN(nr_lattice_features*nr_resolutions_to_slice + 4*nr_resolutions_to_slice,32),
        #     # torch.nn.Softplus(),
        #     torch.nn.Mish(),
        #     torch.nn.Linear(32,32),
        #     # torch.nn.Softplus(),
        #     torch.nn.Mish(),
        #     torch.nn.Linear(32,32),
        #     # torch.nn.Softplus(),
        #     torch.nn.Mish(),
        #     torch.nn.Linear(32,1)
        # )
        # apply_weight_init_fn(self, leaky_relu_init, negative_slope=0.0)
        # leaky_relu_init(self.mlp_sdf[-1], negative_slope=1.0)


        self.dense_grid_weights=torch.nn.Parameter( torch.ones( nr_resolutions  )*1.0, requires_grad=True)

        self.mip_trainable=torch.nn.Parameter( torch.ones(1)*5, requires_grad=True)

        self.trainable_iter=torch.nn.Parameter( torch.ones(1)*1e-5, requires_grad=True)
    
        # def backward_hook(grad):
        #     print("backward hook, grad is ", grad)
        #     grad=50*torch.sign(grad)
        #     return grad
        # self.mip_trainable.register_hook(backward_hook)

    def forward(self, points, ls, iter_nr, use_only_dense_grid):

        # print("iter_nr model", iter_nr)


        assert points.shape[1] == 3, "ray_samples_flat should be nx3"

        # window=self.c2f(iter_nr*10.0001+0.3)
        # window=self.c2f(iter_nr*0.0003+0.3)
        # window=self.c2f(iter_nr*0.0001+0.3)
        # window=self.c2f( map_range_val(iter_nr, 0.0, 25000, 0.3, 1.0   ) )
        window=self.c2f( map_range_val(iter_nr-self.nr_iters_c2f_grid, 0.0, self.nr_iters_for_c2f, 0.0, 1.0   ) )
        # window=self.c2f( map_range_val(iter_nr, 0.0, self.nr_iters_for_c2f, 0.0, 1.0   ) )
        # window=self.c2f( map_range_val(iter_nr, 0.0, self.nr_iters_for_c2f, 0.0, 1.0   ) )
        # print("window",window)
        # window=self.c2f(iter_nr*0.001+0.3)
        # window=self.c2f(iter_nr*0.00003+0.3)
        # window=(window>0.0)*1.0 #we do hard steps so that the network doesnt start learning very high values for the windowed dimensions that start annealing with a low value
        # print("window", window)


        #slice multiple grids
        # ray_features_list=[]
        # print("self.dense_grid_weights",self.dense_grid_weights)
        # for i in range(len(self.grid_3d_list)):
        #     window=self.c2f_grid( map_range_val(iter_nr, 0.0, self.nr_iters_c2f_grid, 0.0, 1.0   ) )
        #     points_scaled=points*2 #scale from [-0.5, 0.5] to [-1, 1]
        #     points_scaled=points_scaled.view(1,1,1,-1,3)
        #     # print("points_scaled min max ", points_scaled.min(), points_scaled.max() )
        #     # values= self.grid_3d_list[i]*(window[i]>0,0).float()*self.dense_grid_weights[i]
        #     values= self.grid_3d_list[i]*  window[i] * self.dense_grid_weights[i]
        #     # ray_features=grid_sample_3d(values, points_scaled)
        #     # ray_features=torch.nn.functional.grid_sample( self.c2f_grid_back_hook(values), points_scaled, mode='bilinear',  align_corners=False)
        #     ray_features=torch.nn.functional.grid_sample( values, points_scaled, mode='bilinear', align_corners=True, padding_mode="zeros")
        #     ray_features=ray_features.transpose(1,4)
        #     # dense_sdf+=ray_features.view(-1,1)
        #     ray_features=ray_features.view(-1,1)
        #     ray_features_list.append(ray_features)
        # dense_sdf=torch.stack(ray_features_list, dim=0).sum(dim=0) #sum all
        
        #do it with mlp
        # ray_features=torch.cat(ray_features_list,1)
        # dense_sdf=self.mlp_sdf_grid(ray_features)



        #slice from multiple grid but it's actually the same one
        # ray_features_list=[]
        # print("self.dense_grid_weights",self.dense_grid_weights)
        # for i in range(len(self.grid_3d_list)):
        #     window=self.c2f_grid( map_range_val(iter_nr, 0.0, self.nr_iters_c2f_grid, 0.0, 1.0   ) )
        #     points_scaled=points*2 #scale from [-0.5, 0.5] to [-1, 1]
        #     points_scaled=points_scaled.view(1,1,1,-1,3)
        #     # print("points_scaled min max ", points_scaled.min(), points_scaled.max() )
        #     # values= self.grid_3d_list[i]*(window[i]>0,0).float()*self.dense_grid_weights[i]
        #     values=  torch.nn.functional.interpolate(self.dense_grid, size=int(self.grid_size_list[i]), mode="nearest")  *  window[i] *self.dense_grid_weights[i]
        #     # ray_features=grid_sample_3d(values, points_scaled)
        #     # ray_features=torch.nn.functional.grid_sample( self.c2f_grid_back_hook(values), points_scaled, mode='bilinear',  align_corners=False)
        #     ray_features=torch.nn.functional.grid_sample( values, points_scaled, mode='bilinear', align_corners=False, padding_mode="zeros")
        #     ray_features=ray_features.transpose(1,4)
        #     # dense_sdf+=ray_features.view(-1,1)
        #     ray_features=ray_features.view(-1,1)
        #     ray_features_list.append(ray_features)
        # dense_sdf=torch.stack(ray_features_list, dim=0).sum(dim=0) #sum all




       


        # #slice 3d grid for dense sdf
        # points_scaled=points*2.0 #scale from [-0.5, 0.5] to [-1, 1]
        # points_scaled=points_scaled.view(1,1,1,-1,3)
        # values= self.dense_grid 
        # ray_features=torch.nn.functional.grid_sample( self.c2f_grid_back_hook(values), points_scaled, mode='bilinear',  align_corners=False)
        # # ray_features=torch.nn.functional.grid_sample( values, points_scaled, mode='bilinear', align_corners=False, padding_mode="zeros")
        # ray_features=ray_features.transpose(1,4)
        # dense_sdf=ray_features.view(-1,1)



        #----------------do it with one grid but with my custom one
        # points_scaled=points*2.0 #scale from [-0.5, 0.5] to [-1, 1]
        # ray_features=self.slice_voxel_grid( self.dense_grid_custom, points_scaled, align_corners=True, iter_nr=iter_nr)
        # dense_sdf=ray_features[:,0:1]

        #slice at two resolutions and blend them together
        # points_scaled=points*2.0 #scale from [-0.5, 0.5] to [-1, 1]
        # cur_mip=4
        # size_for_mip=VoxelGrid.get_size_for_mip(cur_mip, self.dense_grid_size)
        # print("size_for_mip", size_for_mip)
        # ray_features=self.slice_voxel_grid( self.dense_grid_custom, points_scaled, align_corners=True, mip=cur_mip, iter_nr=iter_nr)
        # dense_sdf=ray_features[:,0:1]

        #do it again with a custom one but my lord just put the values at the corners of the voxels and not in the middle
        #slicing 3 times 3 values
        if False:
            points_scaled=points*2.0 #scale from [-0.5, 0.5] to [-1, 1]
            # nr_voxels=129
            # for i in range(8):
                # nr_voxels=VoxelGrid.get_size_downsampled_grid(nr_voxels)
                # nr_voxels=VoxelGrid.get_size_upsampled_grid(nr_voxels)
                # print("nr_voxels", nr_voxels)
            # print("nr mips ", VoxelGrid.get_nr_of_mips(3))
            # ray_features=self.slice_voxel_grid( self.dense_grid_custom, points_scaled, mip=6, iter_nr=iter_nr)
            mip_trainable=torch.clamp(self.mip_trainable, min=1)
            cur_mip=int(torch.round(mip_trainable).item()) #beggining is 5
            prev_mip=cur_mip+1 #beggining is 6
            next_mip=cur_mip-1 
            print("prev_mip", prev_mip)
            print("cur_mip", cur_mip)
            print("next_mip", next_mip)
            print("mip_trainable",mip_trainable)
            ray_features_prev=self.slice_voxel_grid( self.dense_grid_custom, points_scaled, use_nearest=False, mip=prev_mip, iter_nr=iter_nr)
            ray_features_cur=self.slice_voxel_grid( self.dense_grid_custom, points_scaled, use_nearest=False, mip=cur_mip, iter_nr=iter_nr)
            ray_features_next=self.slice_voxel_grid( self.dense_grid_custom, points_scaled, use_nearest=False, mip=next_mip, iter_nr=iter_nr)
            # weight_cur=1.0-torch.frac(mip_trainable)
            #attempt 2 too much clamping so when the distance is above 1 is stops training entirely
            # weight_cur=1.0 - torch.clamp((cur_mip - mip_trainable).abs(),  max=1.0)
            # weight_prev=1.0 - torch.clamp((prev_mip - mip_trainable).abs(),  max=1.0)  +0.001
            # weight_next=1.0 - torch.clamp((next_mip - mip_trainable).abs(),  max=1.0)  +0.001

            #attempt 3 with gaussian it kinda works but the weights need to sum to 1 and so that the eikonal loss can be satisfied at every level
            #i think the gaussian has some weird ups and down in the weights when normalizing, instead they should be monotonically increase with distance
            # mip_dist_to_cur_mip=(cur_mip - mip_trainable).abs()
            # mip_dist_to_prev_mip=(prev_mip - mip_trainable).abs()
            # mip_dist_to_next_mip=(next_mip - mip_trainable).abs()
            # sigma=0.3
            # print("mip_dist_to_cur_mip",mip_dist_to_cur_mip)
            # print("mip_dist_to_prev_mip",mip_dist_to_prev_mip)
            # weight_cur=gauss_activ(mip_dist_to_cur_mip,sigma)
            # weight_prev=gauss_activ(mip_dist_to_prev_mip,sigma)
            # weight_next=gauss_activ(mip_dist_to_next_mip,sigma)
            # weight_sum=weight_cur+weight_prev+weight_next
            # weight_cur=weight_cur/weight_sum
            # weight_prev=weight_prev/weight_sum
            # weight_next=weight_next/weight_sum

            #attemp4 ,we map a distance of 1.5 to weight 0 and a distance of 0.5 to 0.5. this gives rise to the linear equation of w=0.75-0.5distance 
            #wolfram linear regression https://www.wolframalpha.com/widgets/view.jsp?id=a96a9e81ac4bbb54f8002bb61b8d3472
            mip_dist_to_cur_mip=(cur_mip - mip_trainable).abs()
            mip_dist_to_prev_mip=(prev_mip - mip_trainable).abs()
            mip_dist_to_next_mip=(next_mip - mip_trainable).abs()
            # weight_cur=0.75-0.5*mip_dist_to_cur_mip
            weight_prev=0.75-0.5*mip_dist_to_prev_mip
            weight_next=0.75-0.5*mip_dist_to_next_mip
            weight_cur=1.0-weight_prev-weight_next
            weight_sum=weight_cur+weight_prev+weight_next
            # weight_cur=weight_cur/weight_sum
            # weight_prev=weight_prev/weight_sum
            # weight_next=weight_next/weight_sum


            # print("torch.frac(mip_trainable)",torch.frac(mip_trainable))
            # print("mip_trainable-torch.floor(mip_trainable)", mip_trainable-torch.floor(mip_trainable) )
            print("weight_sum", weight_sum)
            print("weight_prev", weight_prev)
            print("weight_cur", weight_cur)
            print("weight_next", weight_next)
            # ray_features=ray_features_prev*weight_prev + ray_features_cur*weight_cur + ray_features_next*weight_next
            ray_features=ray_features_prev*weight_prev.detach() + ray_features_cur*weight_cur.detach() + ray_features_next*weight_next

            dense_sdf=ray_features[:,0:1]

        #Slicing two times
        if False:
            points_scaled=points*2.0 #scale from [-0.5, 0.5] to [-1, 1]

           
            mip_trainable=torch.clamp(self.mip_trainable, min=0)
            prev_mip=int(torch.round(mip_trainable).item()) #beggining is 5
            next_mip=prev_mip-1 
            print("prev_mip", prev_mip)
            print("next_mip", next_mip)
            print("mip_trainable",mip_trainable)
            ray_features_prev=self.slice_voxel_grid( self.dense_grid_custom, points_scaled, use_nearest=False, mip=prev_mip, iter_nr=iter_nr)
            ray_features_next=self.slice_voxel_grid( self.dense_grid_custom, points_scaled, use_nearest=False, mip=next_mip, iter_nr=iter_nr)
           
            #attemp4 ,we map a distance of 1.5 to weight 0 and a distance of 0.5 to 0.5. this gives rise to the linear equation of w=0.75-0.5distance 
            #wolfram linear regression https://www.wolframalpha.com/widgets/view.jsp?id=a96a9e81ac4bbb54f8002bb61b8d3472
            mip_dist_to_prev_mip=(prev_mip - mip_trainable).abs()
            mip_dist_to_next_mip=(next_mip - mip_trainable).abs()
            # weight_cur=0.75-0.5*mip_dist_to_cur_mip
            weight_prev=1.0-mip_dist_to_prev_mip
            weight_next=1.0-mip_dist_to_next_mip
           
            # print("torch.frac(mip_trainable)",torch.frac(mip_trainable))
            # print("mip_trainable-torch.floor(mip_trainable)", mip_trainable-torch.floor(mip_trainable) )
            print("weight_prev", weight_prev)
            print("weight_next", weight_next)
            # ray_features=ray_features_prev*weight_prev + ray_features_next*weight_next
            ray_features=ray_features_prev*weight_prev.detach() + ray_features_next*weight_next

            dense_sdf=ray_features[:,0:1]


        #just do it with c2f but modifying the c2f here
        if False:
            points_scaled=points*2.0 #scale from [-0.5, 0.5] to [-1, 1]
            mip=int(map_range_val(iter_nr, 0.0, self.nr_iters_c2f_grid, 6.9, 0.0   ))
            # mip=5 
            if use_only_dense_grid:
                mip=0
            print("mip is ", mip)
            ray_features=self.slice_voxel_grid( self.dense_grid_custom, points_scaled, use_nearest=False, mip=mip, iter_nr=iter_nr)
            dense_sdf=ray_features[:,0:1]


        #Just do it with the c2f hook and doubel slice 
        #FOR THIS ENABLE C@F
        if False:
            points_scaled=points*2.0 #scale from [-0.5, 0.5] to [-1, 1]
            ray_features=self.slice_voxel_grid( self.dense_grid_custom, points_scaled, use_nearest=False, mip=0, iter_nr=iter_nr)
            dense_sdf=ray_features[:,0:1]

        #fuck it just slice very mip map and splat at every lvl
        if False:
            nr_mips=VoxelGrid.get_nr_of_mips(self.dense_grid_size)-1
            points_scaled=points*2.0 #scale from [-0.5, 0.5] to [-1, 1]
            ray_features_list=[]
            # window_grid_up=self.c2f_grid( map_range_val(iter_nr, 0.0, self.nr_iters_c2f_grid/2, 0.0, 1.0   ) )
            # window_grid_down=self.c2f_grid( map_range_val(iter_nr, self.nr_iters_c2f_grid/2, self.nr_iters_c2f_grid, 0.0, 1.0   ) )
            print("self.dense_grid_weights",self.dense_grid_weights)
            # self.dense_grid_weights.detach()[nr_mips-1]=1.0
            # print("window_grid_up",window_grid_up)
            # print("window_grid_down",window_grid_down)
            window_grid_down_trainable=cosine_easing_window(nr_mips, self.trainable_iter.abs()*nr_mips)
            window_grid_down_trainable=1.0-window_grid_down_trainable
            window_grid_down_trainable[-1]=1.0
            print("window_grid_down_trainable",window_grid_down_trainable)
            print(" self.trainable_iter", self.trainable_iter)
            for i in range(nr_mips):
                mip=nr_mips-i-1
                # print("mip",mip)
                ray_features=self.slice_voxel_grid( self.dense_grid_custom, points_scaled, use_nearest=False, mip=mip, iter_nr=iter_nr)
                # ray_features=ray_features*window[i]*self.dense_grid_weights[i]
                # ray_features=ray_features*window[i]*self.dense_grid_weights[i]
                # w_down=window_grid_down[i]
                # if mip==0 and iter_nr>self.nr_iters_c2f_grid/2:
                    # w_down=0.0
                # w=window_grid_up[i]*(1.0-w_down)
                # w=window_grid_up[i]
                # w=torch.clamp(self.dense_grid_weights[i],min=0.0)
                # print("w",w)
                # augmenter=window_grid_down.sum()
                # print("augmenter",augmenter)
                # ray_features=ray_features*w*(1+augmenter)
                # if window_grid_down[i]>0.0  and  iter_nr>self.nr_iters_c2f_grid/2:
                    # w=w*self.dense_grid_weights[i]

                w=window_grid_down_trainable[i] 
                augmenter=1/window_grid_down_trainable.sum().detach() #if we reduce the sdf of some of them, it will be difficult to hold the eikonal loss, so we augment it so that all the sdfs al always valid
                #Augmenter actually makes it worse
                # print("augmenter",augmenter)
                
                ray_features=ray_features*w*augmenter
                ray_features_list.append(ray_features)
            dense_sdf=torch.stack(ray_features_list, dim=0).sum(dim=0) #sum all

        #forward pass slice from finest lvl and then splat gradient at all mips
        if True:
            points_scaled=points*2.0 #scale from [-0.5, 0.5] to [-1, 1]
            ray_features=self.slice_voxel_grid( self.dense_grid_custom, points_scaled, use_nearest=False, mip=0, iter_nr=iter_nr)
            dense_sdf=ray_features[:,0:1]





        dense_sdf=dense_sdf+self.sdf_shift

        # if use_only_dense_grid:
            # return dense_sdf, None, None
       
       
            
        # ray_features_full, splatting_indices, splatting_weights=self.slice_lattice_monolithic(self.lattice_values_monolithic, self.scale_factor, ls, points, self.random_shift_monolithic.detach(), window.view(-1), concat_points=True, points_scaling=1e-3)
          

       


        
        # x=ray_features_full
        # sdf_and_feat=self.mlp_sdf(x)
        # sdf_residual=sdf_and_feat[:,0:1]


        sdf_residual=None
        feat=None

        #residual
        if use_only_dense_grid:
            # sdf=dense_sdf
            # sdf=ray_features_cur+self.sdf_shift
            ray_features=self.slice_voxel_grid( self.dense_grid_custom, points_scaled, use_nearest=False, mip=0, iter_nr=iter_nr)
            sdf=ray_features[:,0:1]+self.sdf_shift
        else:
            #if are annealing the dense grid, the sparse one should be zero but it is not because the mlp adds some bias, so we make explicitly the sdf as zero


            if iter_nr<self.nr_iters_c2f_grid:
            # if iter_nr<0:
                sdf=dense_sdf
            else:
                ray_features_full, splatting_indices, splatting_weights=self.slice_lattice_monolithic(self.lattice_values_monolithic, self.scale_factor, ls, points, self.random_shift_monolithic.detach(), window.view(-1), concat_points=True, points_scaling=1e-3)

                # ray_features=ray_features[:,-4:]
                # ray_features_full=torch.cat([ray_features_full, ray_features],1)
          
                x=ray_features_full
                sdf_and_feat=self.mlp_sdf(x)
                sdf_residual=sdf_and_feat[:,0:1]


                sdf=sdf_residual+dense_sdf

            ###NOPE no hash at all 
            # sdf=dense_sdf

        if self.feat_size_out!=0: #we should output some feature for the local geometry
            if iter_nr<self.nr_iters_c2f_grid or use_only_dense_grid:
                feat=torch.zeros(points.shape[0],self.feat_size_out)
            else:
                feat=sdf_and_feat[:,-self.feat_size_out:]
        else:
            feat=None
        




        return sdf, feat, sdf_residual

    def get_sdf_and_gradient(self, points, ls, iter_nr, use_only_dense_grid):

        method="finite_difference" #autograd, finite_difference
        # method="autograd" #autograd, finite_difference
        # do_comparison=True #when running finite difference, run also autograd and check that they give rougly the same result
        do_comparison=False #when running finite difference, run also autograd and check that they give rougly the same result

        #dummy
        # sdf, feat = self.forward(points, ls, iter_nr)
        # return sdf, points.clone(), feat

        if method=="finite_difference":
            # do it by finite differences because doing it with autograd doesnt work with the lattice since it would require a double backward through the splatting and slicing function

            with torch.set_grad_enabled(False):
                #to the original positions, add also a tiny epsilon in all directions
                nr_points_original=points.shape[0]
                epsilon=1e-4
                # points_xplus=points.clone()
                # points_yplus=points.clone()
                # points_zplus=points.clone()
                # points_xplus[:,0]=points_xplus[:,0]+epsilon
                # points_yplus[:,1]=points_yplus[:,1]+epsilon
                # points_zplus[:,2]=points_zplus[:,2]+epsilon
                # points_full=torch.cat([points, points_xplus, points_yplus, points_zplus],0)

                eps_x=torch.tensor([epsilon,0,0], dtype=torch.float32, device='cuda:0')
                eps_y=torch.tensor([0,epsilon,0], dtype=torch.float32, device='cuda:0')
                eps_z=torch.tensor([0,0,epsilon], dtype=torch.float32, device='cuda:0')
                points_full=torch.cat([points, points+eps_x, points+eps_y, points+eps_z],0)

            sdf_full, feat_full, sdf_residual_full = self.forward(points_full, ls, iter_nr, use_only_dense_grid)

            feat=None
            if feat_full is not None:            
                feats=feat_full.chunk(4, dim=0) 
                feat=feats[0]

            # dense_sdf=None
            # if dense_sdf_full is not None:
            #     dense_sdfs=dense_sdf_full.chunk(4, dim=0) 
            #     dense_sdf=dense_sdfs[0]

            sdf_residual=None
            if sdf_residual_full is not None:
                sdf_residuals=sdf_residual_full.chunk(4, dim=0) 
                sdf_residual=sdf_residuals[0]

            sdfs=sdf_full.chunk(4, dim=0) 
            sdf=sdfs[0]
            sdf_xplus=sdfs[1]
            sdf_yplus=sdfs[2]
            sdf_zplus=sdfs[3]


            grad_x=(sdf_xplus-sdf)/epsilon
            grad_y=(sdf_yplus-sdf)/epsilon
            grad_z=(sdf_zplus-sdf)/epsilon
            
            gradients=torch.cat([grad_x, grad_y, grad_z],1)


            if do_comparison:
                #do it with autograd
                with torch.set_grad_enabled(True):
                    points.requires_grad_(True)
                    sdf, feat = self.forward(points, ls, iter_nr)
                    # ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
                    # if self.sdf_bounding_sphere > 0.0:
                    #     sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
                    #     sdf = torch.minimum(sdf, sphere_sdf)
                    # feature_vectors = output[:, 1:]

                    feature_vectors=None
                    d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
                    gradients_autograd = torch.autograd.grad(
                        outputs=sdf,
                        inputs=points,
                        grad_outputs=d_output,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True)[0]
                diff=((gradients-gradients_autograd)**2).mean()
                if diff.item()>1e-6:
                    print("error is to high. diff is ", diff)
                    print("gradients finite_diff is ", gradients)
                    print("gradients autograd is ", gradients_autograd)
                    exit(1)



        elif method=="autograd":

            #do it with autograd
            with torch.set_grad_enabled(True):
                points.requires_grad_(True)
                sdf, feat = self.forward(points, ls, iter_nr)
                # ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
                # if self.sdf_bounding_sphere > 0.0:
                #     sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
                #     sdf = torch.minimum(sdf, sphere_sdf)
                # feature_vectors = output[:, 1:]

                feature_vectors=None
                d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
                gradients = torch.autograd.grad(
                    outputs=sdf,
                    inputs=points,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)[0]
        else:
            print("method not known")
            exit(1)




        return sdf, gradients, feat, sdf_residual

    def get_sdf_and_curvature_1d_precomputed_gradient(self, points, sdf_center, sdf_gradients, ls, iter_nr, use_only_dense_grid):
        #get the curvature along a certain random direction for each point

        method="finite_difference" #autograd, finite_difference


        if method=="finite_difference":
            # do it by finite differences because doing it with autograd doesnt work with the lattice since it would require a double backward through the splatting and slicing function
            #do it by taking the central point and then the p+epsilon and p-epsilon in some random directions

            #to the original positions, add also a tiny epsilon 
            nr_points_original=points.shape[0]
            epsilon=1e-4
            # epsilon_end=1e-4
            # epsilon=map_range_val(iter_nr, 0, 10000, 1e-2, epsilon_end)
            # epsilon=np.random.uniform(1e-4,1e-3)
            rand_directions=torch.randn_like(points)
            rand_directions=F.normalize(rand_directions,dim=-1)

            #instead of random direction we take the normals at these points, and calculate a random vector that is orthogonal 
            # with torch.set_grad_enabled(True):
            #     sdf, sdf_gradients, feat=self.get_sdf_and_gradient(points, ls, iter_nr)
            #     normals=F.normalize(sdf_gradients,dim=-1)
            #     tangent=torch.cross(normals, rand_directions)
            #     rand_directions=tangent #set the random moving direction to be the tangent direction now

            normals=F.normalize(sdf_gradients,dim=-1)
            # normals=normals.detach()
            tangent=torch.cross(normals, rand_directions)
            rand_directions=tangent #set the random moving direction to be the tangent direction now

            

            points_plus=points.clone()+rand_directions*epsilon
            points_minus=points.clone()-rand_directions*epsilon

            points_full=torch.cat([ points_plus, points_minus],0)

            sdf_full, feat_full, sdf_residual = self.forward(points_full, ls, iter_nr, use_only_dense_grid)


            #we dont return a feat because we don't compute here the feat of the center point
            # feat=None
            # if feat_full is not None:            
                # feats=feat_full.chunk(2, dim=0) 
                # feat=feats[0]

            sdfs=sdf_full.chunk(2, dim=0) 
            sdf=sdf_center
            sdf_plus=sdfs[0]
            sdf_minus=sdfs[1]

            curvature=(sdf_plus-2*sdf+sdf_minus)/(epsilon*epsilon)
            


           

        else:
            print("method not known")
            exit(1)




        return sdf, curvature, None, None #we dont return a feat because we don't compute here the feat of the center point

    def get_sdf_and_curvature_1d(self, points, ls, iter_nr):
        #get the curvature along a certain random direction for each point

        method="finite_difference" #autograd, finite_difference


        if method=="finite_difference":
            # do it by finite differences because doing it with autograd doesnt work with the lattice since it would require a double backward through the splatting and slicing function
            #do it by taking the central point and then the p+epsilon and p-epsilon in some random directions

            #to the original positions, add also a tiny epsilon 
            nr_points_original=points.shape[0]
            epsilon=1e-4
            rand_directions=torch.randn_like(points)
            rand_directions=F.normalize(rand_directions,dim=-1)

            # instead of random direction we take the normals at these points, and calculate a random vector that is orthogonal 
            with torch.set_grad_enabled(True):
                sdf, sdf_gradients, feat=self.get_sdf_and_gradient(points, ls, iter_nr)
                normals=F.normalize(sdf_gradients,dim=-1)
                tangent=torch.cross(normals, rand_directions)
                rand_directions=tangent #set the random moving direction to be the tangent direction now

            normals=F.normalize(sdf_gradients,dim=-1)
            tangent=torch.cross(normals, rand_directions)
            rand_directions=tangent #set the random moving direction to be the tangent direction now

            

            points_plus=points.clone()+rand_directions*epsilon
            points_minus=points.clone()-rand_directions*epsilon

            points_full=torch.cat([points, points_plus, points_minus],0)

            sdf_full, feat_full = self.forward(points_full, ls, iter_nr)

            feat=None
            if feat_full is not None:            
                feats=feat_full.chunk(3, dim=0) 
                feat=feats[0]

            sdfs=sdf_full.chunk(3, dim=0) 
            sdf=sdfs[0]
            sdf_plus=sdfs[1]
            sdf_minus=sdfs[2]

            curvature=(sdf_plus-2*sdf+sdf_minus)/(epsilon*epsilon)
            


           

        else:
            print("method not known")
            exit(1)




        return sdf, curvature, feat

    def get_sdf_and_curvature_1d_precomputed_gradient_normal_based(self, points, sdf_gradients, ls, iter_nr):
        #get the curvature along a certain random direction for each point
        #does it by computing the normal at a shifted point on the tangent plant and then computing a dot produt

        method="finite_difference" #autograd, finite_difference


        if method=="finite_difference":
            # do it by finite differences because doing it with autograd doesnt work with the lattice since it would require a double backward through the splatting and slicing function
            #do it by taking the central point and then the p+epsilon and p-epsilon in some random directions

            #to the original positions, add also a tiny epsilon 
            nr_points_original=points.shape[0]
            epsilon=1e-4
            # epsilon=np.random.uniform(1e-4,1e-3)
            rand_directions=torch.randn_like(points)
            rand_directions=F.normalize(rand_directions,dim=-1)

            #instead of random direction we take the normals at these points, and calculate a random vector that is orthogonal 
            # with torch.set_grad_enabled(True):
            #     sdf, sdf_gradients, feat=self.get_sdf_and_gradient(points, ls, iter_nr)
            #     normals=F.normalize(sdf_gradients,dim=-1)
            #     tangent=torch.cross(normals, rand_directions)
            #     rand_directions=tangent #set the random moving direction to be the tangent direction now

            normals=F.normalize(sdf_gradients,dim=-1)
            # normals=normals.detach()
            tangent=torch.cross(normals, rand_directions)
            rand_directions=tangent #set the random moving direction to be the tangent direction now

            

            points_shifted=points.clone()+rand_directions*epsilon
            
            #get the gradient at the shifted point
            sdf_shifted, sdf_gradients_shifted, feat_shifted=self.get_sdf_and_gradient(points_shifted, ls, iter_nr) 

            normals_shifted=F.normalize(sdf_gradients_shifted,dim=-1)

            dot=(normals*normals_shifted).sum(dim=-1, keepdim=True)
            curvature=1-dot

            # curvature= ((normals-normals_shifted)**2 ).sum(dim=-1, keepdim=True)



           

        else:
            print("method not known")
            exit(1)




        return sdf_shifted, curvature, feat_shifted

    def save(self, root_folder, experiment_name, iter_nr):

        models_path=os.path.join(root_folder,"checkpoints/", experiment_name, str(iter_nr), "models")
        os.makedirs(models_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(models_path, "sdf_model.pt")  )

class SDF(torch.nn.Module):

    def __init__(self, in_channels, boundary_primitive, geom_feat_size_out, nr_iters_for_c2f):
        super(SDF, self).__init__()

        self.in_channels=in_channels
        self.boundary_primitive=boundary_primitive
        self.geom_feat_size_out=geom_feat_size_out


        #create encoding
        pos_dim=in_channels
        capacity=pow(2,18) #2pow18
        nr_levels=24 
        nr_feat_per_level=2 
        coarsest_scale=1.0 
        finest_scale=0.0001 
        scale_list=np.geomspace(coarsest_scale, finest_scale, num=nr_levels)
        self.encoding=permuto_enc.PermutoEncoding(pos_dim, capacity, nr_levels, nr_feat_per_level, scale_list, appply_random_shift_per_level=True, concat_points=True, concat_points_scaling=1e-3)           

        
        self.sdf_shift=1e-2
        self.mlp_sdf= torch.nn.Sequential(
            torch.nn.Linear(self.encoding.output_dims() ,32),
            torch.nn.GELU(),
            torch.nn.Linear(32,32),
            torch.nn.GELU(),
            torch.nn.Linear(32,32),
            torch.nn.GELU(),
            torch.nn.Linear(32,1+geom_feat_size_out)
        )
        apply_weight_init_fn(self.mlp_sdf, leaky_relu_init, negative_slope=0.0)
        leaky_relu_init(self.mlp_sdf[-1], negative_slope=1.0)
        with torch.set_grad_enabled(False):
            self.mlp_sdf[-1].bias+=self.sdf_shift #faster if we just put it in the bias

       


        self.c2f=permuto_enc.Coarse2Fine(nr_levels)
        self.nr_iters_for_c2f=nr_iters_for_c2f
        self.last_iter_nr=sys.maxsize

    def forward(self, points, iter_nr):

        assert points.shape[1] == self.in_channels, "points should be N x in_channels"

        self.last_iter_nr=iter_nr

       
        window=self.c2f( map_range_val(iter_nr, 0.0, self.nr_iters_for_c2f, 0.3, 1.0   ) )

     
        point_features=self.encoding(points, window.view(-1))
        sdf_and_feat=self.mlp_sdf(point_features)
        
        if self.geom_feat_size_out!=0:
            sdf=sdf_and_feat[:,0:1]
            geom_feat=sdf_and_feat[:,-self.geom_feat_size_out:]
        else:
            sdf=sdf_and_feat
            geom_feat=None


        return sdf, geom_feat

    def get_sdf_and_gradient(self, points, iter_nr, method="autograd"):


        if method=="finite_difference":
            with torch.set_grad_enabled(False):
                #to the original positions, add also a tiny epsilon in all directions
                nr_points_original=points.shape[0]
                epsilon=1e-4
                points_xplus=points.clone()
                points_yplus=points.clone()
                points_zplus=points.clone()
                points_xplus[:,0]=points_xplus[:,0]+epsilon
                points_yplus[:,1]=points_yplus[:,1]+epsilon
                points_zplus[:,2]=points_zplus[:,2]+epsilon
                points_full=torch.cat([points, points_xplus, points_yplus, points_zplus],0)

               
            sdf_full, geom_feat_full = self.forward(points_full, iter_nr)

            geom_feat=None
            if geom_feat_full is not None:            
                g_feats=geom_feat_full.chunk(4, dim=0) 
                geom_feat=g_feats[0]

            sdfs=sdf_full.chunk(4, dim=0) 
            sdf=sdfs[0]
            sdf_xplus=sdfs[1]
            sdf_yplus=sdfs[2]
            sdf_zplus=sdfs[3]

            grad_x=(sdf_xplus-sdf)/epsilon
            grad_y=(sdf_yplus-sdf)/epsilon
            grad_z=(sdf_zplus-sdf)/epsilon

            gradients=torch.cat([grad_x, grad_y, grad_z],1)


        elif method=="autograd":

            #do it with autograd
            with torch.set_grad_enabled(True):
                points.requires_grad_(True)
                sdf, geom_feat = self.forward(points, iter_nr)

                feature_vectors=None
                d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
                gradients = torch.autograd.grad(
                    outputs=sdf,
                    inputs=points,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)[0]
        else:
            print("method not known")
            exit(1)




        return sdf, gradients, geom_feat

    # def get_sdf_and_curvature_1d_precomputed_gradient(self, points, sdf_center, sdf_gradients, ls, iter_nr, use_only_dense_grid):
    #     #get the curvature along a certain random direction for each point

    #     # method="finite_difference" #autograd, finite_difference


    #     if method=="finite_difference":
    #         # do it by finite differences because doing it with autograd doesnt work with the lattice since it would require a double backward through the splatting and slicing function
    #         #do it by taking the central point and then the p+epsilon and p-epsilon in some random directions

    #         #to the original positions, add also a tiny epsilon 
    #         nr_points_original=points.shape[0]
    #         epsilon=1e-4
    #         # epsilon_end=1e-4
    #         # epsilon=map_range_val(iter_nr, 0, 10000, 1e-2, epsilon_end)
    #         # epsilon=np.random.uniform(1e-4,1e-3)
    #         rand_directions=torch.randn_like(points)
    #         rand_directions=F.normalize(rand_directions,dim=-1)

    #         #instead of random direction we take the normals at these points, and calculate a random vector that is orthogonal 
    #         # with torch.set_grad_enabled(True):
    #         #     sdf, sdf_gradients, feat=self.get_sdf_and_gradient(points, ls, iter_nr)
    #         #     normals=F.normalize(sdf_gradients,dim=-1)
    #         #     tangent=torch.cross(normals, rand_directions)
    #         #     rand_directions=tangent #set the random moving direction to be the tangent direction now

    #         normals=F.normalize(sdf_gradients,dim=-1)
    #         # normals=normals.detach()
    #         tangent=torch.cross(normals, rand_directions)
    #         rand_directions=tangent #set the random moving direction to be the tangent direction now

            

    #         points_plus=points.clone()+rand_directions*epsilon
    #         points_minus=points.clone()-rand_directions*epsilon

    #         points_full=torch.cat([ points_plus, points_minus],0)

    #         sdf_full, feat_full, sdf_residual = self.forward(points_full, ls, iter_nr, use_only_dense_grid)


    #         #we dont return a feat because we don't compute here the feat of the center point
    #         # feat=None
    #         # if feat_full is not None:            
    #             # feats=feat_full.chunk(2, dim=0) 
    #             # feat=feats[0]

    #         sdfs=sdf_full.chunk(2, dim=0) 
    #         sdf=sdf_center
    #         sdf_plus=sdfs[0]
    #         sdf_minus=sdfs[1]

    #         curvature=(sdf_plus-2*sdf+sdf_minus)/(epsilon*epsilon)
            


           

    #     else:
    #         print("method not known")
    #         exit(1)




    #     return sdf, curvature, None, None #we dont return a feat because we don't compute here the feat of the center point

    # def get_sdf_and_curvature_1d(self, points, ls, iter_nr):
    #     #get the curvature along a certain random direction for each point

    #     method="finite_difference" #autograd, finite_difference


    #     if method=="finite_difference":
    #         # do it by finite differences because doing it with autograd doesnt work with the lattice since it would require a double backward through the splatting and slicing function
    #         #do it by taking the central point and then the p+epsilon and p-epsilon in some random directions

    #         #to the original positions, add also a tiny epsilon 
    #         nr_points_original=points.shape[0]
    #         epsilon=1e-4
    #         rand_directions=torch.randn_like(points)
    #         rand_directions=F.normalize(rand_directions,dim=-1)

    #         # instead of random direction we take the normals at these points, and calculate a random vector that is orthogonal 
    #         with torch.set_grad_enabled(True):
    #             sdf, sdf_gradients, feat=self.get_sdf_and_gradient(points, ls, iter_nr)
    #             normals=F.normalize(sdf_gradients,dim=-1)
    #             tangent=torch.cross(normals, rand_directions)
    #             rand_directions=tangent #set the random moving direction to be the tangent direction now

    #         normals=F.normalize(sdf_gradients,dim=-1)
    #         tangent=torch.cross(normals, rand_directions)
    #         rand_directions=tangent #set the random moving direction to be the tangent direction now

            

    #         points_plus=points.clone()+rand_directions*epsilon
    #         points_minus=points.clone()-rand_directions*epsilon

    #         points_full=torch.cat([points, points_plus, points_minus],0)

    #         sdf_full, feat_full = self.forward(points_full, ls, iter_nr)

    #         feat=None
    #         if feat_full is not None:            
    #             feats=feat_full.chunk(3, dim=0) 
    #             feat=feats[0]

    #         sdfs=sdf_full.chunk(3, dim=0) 
    #         sdf=sdfs[0]
    #         sdf_plus=sdfs[1]
    #         sdf_minus=sdfs[2]

    #         curvature=(sdf_plus-2*sdf+sdf_minus)/(epsilon*epsilon)
            


           

    #     else:
    #         print("method not known")
    #         exit(1)




    #     return sdf, curvature, feat

    def get_sdf_and_curvature_1d_precomputed_gradient_normal_based(self, points, sdf_gradients,iter_nr):
        #get the curvature along a certain random direction for each point
        #does it by computing the normal at a shifted point on the tangent plant and then computing a dot produt



        #to the original positions, add also a tiny epsilon 
        nr_points_original=points.shape[0]
        epsilon=1e-4
        rand_directions=torch.randn_like(points)
        rand_directions=F.normalize(rand_directions,dim=-1)

        #instead of random direction we take the normals at these points, and calculate a random vector that is orthogonal 
        normals=F.normalize(sdf_gradients,dim=-1)
        # normals=normals.detach()
        tangent=torch.cross(normals, rand_directions)
        rand_directions=tangent #set the random moving direction to be the tangent direction now
        

        points_shifted=points.clone()+rand_directions*epsilon
        
        #get the gradient at the shifted point
        sdf_shifted, sdf_gradients_shifted, feat_shifted=self.get_sdf_and_gradient(points_shifted, iter_nr) 

        normals_shifted=F.normalize(sdf_gradients_shifted,dim=-1)

        dot=(normals*normals_shifted).sum(dim=-1, keepdim=True)
        #the dot would assign low weight importance to normals that are almost the same, and increasing error the more they deviate. So it's something like and L2 loss. But we want a L1 loss so we get the angle, and then we map it to range [0,1]
        angle=torch.acos(torch.clamp(dot, -1.0+1e-6, 1.0-1e-6)) #goes to range 0 when the angle is the same and 2pi when is opposite


        curvature=angle/(2.0*math.pi) #map to [0,1 range]

        return sdf_shifted, curvature

    def path_to_save_model(self, ckpt_folder, experiment_name, iter_nr):
        models_path=os.path.join(ckpt_folder, experiment_name, str(iter_nr), "models")
        return models_path

    def save(self, ckpt_folder, experiment_name, iter_nr):

        # models_path=os.path.join(ckpt_folder, experiment_name, str(iter_nr), "models")
        models_path=self.path_to_save_model(ckpt_folder, experiment_name, iter_nr)
        os.makedirs(models_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(models_path, "sdf_model.pt")  )

        return models_path 

class RGB(torch.nn.Module):

    def __init__(self, in_channels, boundary_primitive, geom_feat_size_in, nr_iters_for_c2f):
        super(RGB, self).__init__()

        self.in_channels=in_channels
        self.boundary_primitive=boundary_primitive
        self.geom_feat_size_in=geom_feat_size_in


        # self.pick_rand_rows= RandRowPicker()
        # self.pick_rand_pixels= RandPixelPicker(low_discrepancy=False) #do NOT use los discrepancy for now, it seems to align some of the rays to some directions so maybe it's not that good of an idea
        # self.pixel_sampler=PixelSampler()
        self.create_rays=CreateRaysModule()
        self.volume_renderer_neus = VolumeRenderingNeus()

        #create encoding
        pos_dim=in_channels
        capacity=pow(2,18) #2pow18
        nr_levels=24 
        nr_feat_per_level=2 
        coarsest_scale=1.0 
        finest_scale=0.0001 
        scale_list=np.geomspace(coarsest_scale, finest_scale, num=nr_levels)
        self.encoding=permuto_enc.PermutoEncoding(pos_dim, capacity, nr_levels, nr_feat_per_level, scale_list, appply_random_shift_per_level=True, concat_points=True, concat_points_scaling=1)           

       

        # with dirs encoded
        # self.mlp= torch.nn.Sequential(
        #     torch.nn.Linear(self.encoding.output_dims() + 25 + 3 + geom_feat_size_in, 128),
        #     torch.nn.GELU(),
        #     torch.nn.Linear(128,128),
        #     torch.nn.GELU(),
        #     torch.nn.Linear(128,64),
        #     torch.nn.GELU(),
        #     torch.nn.Linear(64,3)
        # )
        # apply_weight_init_fn(self.mlp, leaky_relu_init, negative_slope=0.0)
        # leaky_relu_init(self.mlp[-1], negative_slope=1.0)
        mlp_in_channels=self.encoding.output_dims() + 25 + 3 + geom_feat_size_in
        self.mlp=LipshitzMLP(mlp_in_channels, [128,128,64,3], last_layer_linear=True)

        self.c2f=permuto_enc.Coarse2Fine(nr_levels)
        self.nr_iters_for_c2f=nr_iters_for_c2f
        self.last_iter_nr=sys.maxsize 

        self.softplus=torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points, samples_dirs, sdf_gradients, geom_feat,  iter_nr, model_colorcal=None, img_indices=None, ray_start_end_idx=None):

     

        assert points.shape[1] == self.in_channels, "points should be N x in_channels"

        self.last_iter_nr=iter_nr

       
        window=self.c2f( map_range_val(iter_nr, 0.0, self.nr_iters_for_c2f, 0.3, 1.0   ) )

        point_features=self.encoding(points, window.view(-1))
        #dirs encoded with spherical harmonics 
        with torch.set_grad_enabled(False):
            samples_dirs_enc=HashSDF.spherical_harmonics(samples_dirs,5)
        #normals
        normals=F.normalize( sdf_gradients.view(-1,3), dim=1 )

       
        x=torch.cat([point_features, samples_dirs_enc, normals, geom_feat],1)
       

        x=self.mlp(x)

        if model_colorcal is not None:
            x=model_colorcal.calib_RGB_samples_packed(x, img_indices, ray_start_end_idx )
        

        x = self.sigmoid(x)
        


        return x

    def path_to_save_model(self, ckpt_folder, experiment_name, iter_nr):
        models_path=os.path.join(ckpt_folder, experiment_name, str(iter_nr), "models")
        return models_path

    def save(self, ckpt_folder, experiment_name, iter_nr):

        # models_path=os.path.join(ckpt_folder, experiment_name, str(iter_nr), "models")
        models_path=self.path_to_save_model(ckpt_folder, experiment_name, iter_nr)
        os.makedirs(models_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(models_path, "rgb_model.pt")  )

        return models_path 

    def parameters_only_encoding(self):
        params=[]
        for name, param in self.encoding.named_parameters():
            if "lattice_values" in name:
                params.append(param)
        return params

    def parameters_all_without_encoding(self):
        params=[]
        for name, param in self.named_parameters():
            if "lattice_values" in name:
                pass
            else:
                params.append(param)
        return params
                


###################NERF ################################
class NerfHash(torch.nn.Module):

    def __init__(self, in_channels, boundary_primitive, nr_iters_for_c2f):
        super(NerfHash, self).__init__()

        self.in_channels=in_channels
        self.boundary_primitive=boundary_primitive


        #create encoding
        pos_dim=in_channels
        capacity=pow(2,18) #2pow18
        nr_levels=24 
        nr_feat_per_level=2 
        coarsest_scale=1.0 
        finest_scale=0.0001 
        scale_list=np.geomspace(coarsest_scale, finest_scale, num=nr_levels)
        self.encoding=permuto_enc.PermutoEncoding(pos_dim, capacity, nr_levels, nr_feat_per_level, scale_list, appply_random_shift_per_level=True, concat_points=True, concat_points_scaling=1)       





        # self.pick_rand_rows= RandRowPicker()
        # self.pick_rand_pixels= RandPixelPicker(low_discrepancy=False) #do NOT use los discrepancy for now, it seems to align some of the rays to some directions so maybe it's not that good of an idea
        # self.pick_patch_pixels= PatchPixelPicker()
        # self.pick_patches_pixels= PatchesPixelPicker()
        # self.pick_patch_and_rand_pixels= PatchAndRandPixelPicker(low_discrepancy=False)
        # self.pixel_sampler=PixelSampler()
        self.create_rays=CreateRaysModule()
        # self.ray_sampler=NerfUniformSampler(boundary_primitive)
        # self.ray_sampler_bg=NerfBGSampler(boundary_primitive, nr_samples_per_ray)
        # self.volume_renderer=VolumeRenderingNerf()
        # self.volume_renderer_general=VolumeRenderingGeneralModule()
        self.volume_renderer_nerf=VolumeRenderingNerf()

        

        self.nr_feat_for_rgb=64
        self.mlp_feat_and_density= torch.nn.Sequential(
            torch.nn.Linear(self.encoding.output_dims(), 64),
            torch.nn.GELU(),
            torch.nn.Linear(64,64),
            torch.nn.GELU(),
            torch.nn.Linear(64,64),
            torch.nn.GELU(),
            torch.nn.Linear(64,self.nr_feat_for_rgb+1) 
        )
        apply_weight_init_fn(self.mlp_feat_and_density, leaky_relu_init, negative_slope=0.0)
       
       

        self.mlp_rgb= torch.nn.Sequential(
            torch.nn.Linear(self.nr_feat_for_rgb+16, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64,3),
        )
        apply_weight_init_fn(self.mlp_rgb, leaky_relu_init, negative_slope=0.0)
        leaky_relu_init(self.mlp_rgb[-1], negative_slope=1.0)
        


        self.softplus=torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()
        self.gelu=torch.nn.GELU()



        self.c2f=permuto_enc.Coarse2Fine(nr_levels)
        self.nr_iters_for_c2f=nr_iters_for_c2f
        self.last_iter_nr=sys.maxsize

    
    # def forward(self, ray_origins, ray_dirs, ls, iter_nr, nr_samples_per_ray):
    def forward(self, samples_pos, samples_dirs, iter_nr, model_colorcal=None, img_indices=None, ray_start_end_idx=None):


        assert samples_pos.shape[1] == self.in_channels, "points should be N x in_channels"

        self.last_iter_nr=iter_nr

       
        window=self.c2f( map_range_val(iter_nr, 0.0, self.nr_iters_for_c2f, 0.3, 1.0   ) ) #helps to converge the radiance of the object in the center and not put radiance on the walls of the bounding box
        

        points=samples_pos
        point_features=self.encoding(points, window.view(-1)) 
        #dirs encoded with spherical harmonics 
        with torch.set_grad_enabled(False):
            samples_dirs_enc=HashSDF.spherical_harmonics(samples_dirs,4)



        #predict density without using directions
        feat_and_density=self.mlp_feat_and_density(point_features)
        density=feat_and_density[:,0:1]
        feat_rgb=feat_and_density[:,1:self.nr_feat_for_rgb+1]

        #predict rgb using directions of ray
        feat_rgb_with_dirs=torch.cat([ self.gelu(feat_rgb), samples_dirs_enc],1)
        rgb=self.mlp_rgb(feat_rgb_with_dirs)

        #activate
        density=self.softplus(density) #similar to mipnerf


        if model_colorcal is not None:
            rgb=model_colorcal.calib_RGB_samples_packed(rgb, img_indices, ray_start_end_idx )

        rgb=self.sigmoid(rgb)


        return rgb, density

    def get_only_density(self, ray_samples, iter_nr):


        # window=self.c2f(iter_nr*0.0001) #helps to converge the radiance of the object in the center and not put radiance on the walls of the bounding box
        window=self.c2f( map_range_val(iter_nr, 0.0, self.nr_iters_for_c2f, 0.3, 1.0   ) ) #helps to converge the radiance of the object in the center and not 



        #given the rays, create points
        points=ray_samples.view(-1,ray_samples.shape[-1])

      


        point_features=self.encoding(points, window.view(-1))  


        #predict density without using directions
        feat_and_density=self.mlp_feat_and_density(point_features)
        density=feat_and_density[:,0:1]
        density=self.softplus(density) #similar to mipnerf

        return density

    def path_to_save_model(self, ckpt_folder, experiment_name, iter_nr):
        models_path=os.path.join(ckpt_folder, experiment_name, str(iter_nr), "models")
        return models_path

    def save(self, ckpt_folder, experiment_name, iter_nr, additional_name=None):

        # models_path=os.path.join(ckpt_folder, experiment_name, str(iter_nr), "models")
        models_path=self.path_to_save_model(ckpt_folder, experiment_name, iter_nr)
        os.makedirs(models_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(models_path, "nerf_hash_model"+str(additional_name or '')+".pt")  )

        return models_path

class Nerf(torch.nn.Module):

    def __init__(self, input_channels, nr_frequencies, boundary_primitive, nr_samples_per_ray):
        super(Nerf, self).__init__()




        # self.pick_rand_rows= RandRowPicker()
        self.pick_rand_pixels= RandPixelPicker(low_discrepancy=False) #do NOT use los discrepancy for now, it seems to align some of the rays to some directions so maybe it's not that good of an idea
        self.pick_patch_pixels= PatchPixelPicker()
        self.pick_patches_pixels= PatchesPixelPicker()
        self.pick_patch_and_rand_pixels= PatchAndRandPixelPicker(low_discrepancy=False)
        self.pixel_sampler=PixelSampler()
        self.create_rays=CreateRaysModule()
        self.ray_sampler=NerfUniformSampler(boundary_primitive, nr_samples_per_ray)
        self.ray_sampler_bg=NerfBGSampler(boundary_primitive, nr_samples_per_ray)
        # self.ray_sampler=NerfSampler(boundary_primitive, N_initial_samples=64, N_samples_importance=64)
        self.volume_renderer=VolumeRenderingNerf()

        self.pos_encode=PositionalEncoding(input_channels, nr_frequencies, only_sin=False) 
        channels_after_encoding=input_channels+input_channels*2*nr_frequencies

        self.nr_feat_for_rgb=128
        self.mlp_feat_and_density= nn.Sequential(
            torch.nn.Linear(channels_after_encoding, 128),
            torch.nn.Mish(),
            torch.nn.Linear(128,128),
            torch.nn.Mish(),
            torch.nn.Linear(128,128),
            torch.nn.Mish(),
            torch.nn.Linear(128,128),
            torch.nn.Mish(),
            torch.nn.Linear(128,128),
            torch.nn.Mish(),
            torch.nn.Linear(128,self.nr_feat_for_rgb+1) 
        )
        apply_weight_init_fn(self.mlp_feat_and_density, leaky_relu_init, negative_slope=0.0)
        leaky_relu_init(self.mlp_feat_and_density[-1], negative_slope=1.0)

       

        self.mlp_rgb= nn.Sequential(
            torch.nn.Linear(self.nr_feat_for_rgb+16, 64),
            torch.nn.Mish(),
            torch.nn.Linear(64, 64),
            torch.nn.Mish(),
            torch.nn.Linear(64,3),
        )
        apply_weight_init_fn(self.mlp_rgb, leaky_relu_init, negative_slope=0.0)
        leaky_relu_init(self.mlp_rgb[-1], negative_slope=1.0)
        


        self.rgb_padding = 0.001  # Padding added to the RGB outputs.


        self.softplus=torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()




    
    # def forward(self, ray_origins, ray_dirs, ls, iter_nr, nr_samples_per_ray):
    def forward(self, ray_samples, ray_dirs, ls, iter_nr):

        # assert points.shape[1] == 3, "ray_samples_flat should be nx3"

        # window=self.c2f(iter_nr*0.0001) #helps to converge the radiance of the object in the center and not put radiance on the walls of the bounding box
        # window=(window>0.0)*1.0
        # print("window", window)



        #given the rays, create points
        nr_rays=ray_samples.shape[0]
        nr_samples_per_ray=ray_samples.shape[1]
        samples_dir=ray_samples.shape[2] #should usually be 3 but it can be 4 when we use it as a nerf++
        points=ray_samples.view(-1,samples_dir)
        nr_times_to_repeat=int(points.shape[0]/ray_dirs.view(-1,3).shape[0])

        # if iter_nr%100==0:
            # show_points(points,"points")



        

        #dirs encoded with spherical harmonics 
        with torch.set_grad_enabled(False):
            ray_dirs_enc=InstantNGP.spherical_harmonics(ray_dirs,4)
            ray_dirs_enc_repeted=ray_dirs_enc.view(-1,1,16).repeat(1,nr_times_to_repeat,1).contiguous().view(-1,16)


        # ray_features_full=self.pos_encode(points).view(nr_rays, nr_samples_per_ray, -1)
        ray_features_full=self.pos_encode(points)
       

        #predict density without using directions
        feat_and_density=self.mlp_feat_and_density(ray_features_full)
        density=feat_and_density[:,0:1]
        feat_rgb=feat_and_density[:,1:self.nr_feat_for_rgb+1]

        #predict rgb using directions of ray
        feat_rgb_with_dirs=torch.cat([feat_rgb, ray_dirs_enc_repeted],1)
        rgb=self.mlp_rgb(feat_rgb_with_dirs)


        #attemtp 2 
        density=density.view(nr_rays, nr_samples_per_ray, 1)
        rgb=rgb.view(nr_rays, nr_samples_per_ray, 3)
        #activate
        density=self.softplus(density) #similar to mipnerf
        rgb=self.sigmoid(rgb)
        #concat
        rgba_field=torch.cat([rgb,density],2) #nr_rays,nr_samples,4




        # return pred_rgb
        return rgba_field

class INGP(torch.nn.Module):

    def __init__(self, input_channels, nr_lattice_vertices, nr_lattice_features, nr_resolutions, boundary_primitive, nr_iters_for_c2f):
        super(INGP, self).__init__()

        self.input_channels=input_channels


        self.pick_rand_pixels= RandPixelPicker(low_discrepancy=False) #do NOT use los discrepancy for now, it seems to align some of the rays to some directions so maybe it's not that good of an idea
        self.create_rays=CreateRaysModule()
        self.volume_renderer_general=VolumeRenderingGeneralModule() #This fully fused rendering of nerf doesn't propagate graditn wrt to the sample weight and therefore cannot be used with mask loss
        self.volume_renderer_nerf=VolumeRenderingNerf()
        self.sum_over_ray=SumOverRayModule()
        self.integrator_module=IntegrateColorAndWeightsModule()


        self.nr_lattice_features=nr_lattice_features
        self.nr_resolutions=nr_resolutions
        


        config_encoding={
            "otype": "HashGrid",
            "n_levels": nr_resolutions,
            "n_features_per_level": nr_lattice_features,
            "log2_hashmap_size": 18,
            "base_resolution": 4
            # "interpolation": "Linear"
        }
        self.encoding = tcnn.Encoding(n_input_dims=input_channels, encoding_config=config_encoding)
    

       

        self.nr_feat_for_rgb=64
        self.mlp_feat_and_density= nn.Sequential(
            torch.nn.Linear(nr_lattice_features*nr_resolutions, 64),
            # torch.nn.Mish(), # DO NOT USE MIsh, the lattice values can be qutie high and the exp here overflows
            # torch.nn.SiLU(),
            torch.nn.ReLU(), 
            # torch.nn.GELU(), 
            torch.nn.Linear(64,64),
            # torch.nn.Mish(),
            # torch.nn.SiLU(),
            torch.nn.ReLU(), 
            # torch.nn.GELU(),
            torch.nn.Linear(64,64),
            # torch.nn.Mish(),
            # torch.nn.SiLU(),
            torch.nn.ReLU(), 
            # torch.nn.GELU(),
            torch.nn.Linear(64,self.nr_feat_for_rgb+1) 
        )
        apply_weight_init_fn(self.mlp_feat_and_density, leaky_relu_init, negative_slope=0.0)

       

        self.mlp_rgb= nn.Sequential(
            torch.nn.Linear(self.nr_feat_for_rgb+16, 64),
            torch.nn.ReLU(),
            # torch.nn.GELU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            # torch.nn.GELU(),
            torch.nn.Linear(64,3),
        )
        apply_weight_init_fn(self.mlp_rgb, leaky_relu_init, negative_slope=0.0)
        leaky_relu_init(self.mlp_rgb[-1], negative_slope=1.0)
        


        self.sigmoid = torch.nn.Sigmoid()
        self.mish=torch.nn.Mish()
        self.softplus=torch.nn.Softplus()



        self.c2f=Coarse2Fine(self.nr_resolutions)
        self.nr_iters_for_c2f=nr_iters_for_c2f

    
    def forward(self, samples_pos, samples_dirs, ls, iter_nr, model_colorcal=None, img_indices=None, ray_start_end_idx=None, nr_rays=None):


        window=self.c2f( map_range_val(iter_nr, 0.0, self.nr_iters_for_c2f, 0.3, 1.0   ) ) #helps to converge the radiance of the object in the center and not put radiance on the walls of the bounding box
       

        #dirs encoded with spherical harmonics 
        TIME_START("encode_dirs")
        with torch.set_grad_enabled(False):
            samples_dirs_enc=InstantNGP.spherical_harmonics(samples_dirs,4)
        TIME_END("encode_dirs")

        ray_features_full=self.encoding(samples_pos) 
        # ray_features_full=ray_features_full.float()
        # print("ray_features_full",ray_features_full.type())
       

        #predict density without using directions
        feat_and_density=self.mlp_feat_and_density(ray_features_full)
        density=feat_and_density[:,0:1]
        feat_rgb=feat_and_density[:,1:self.nr_feat_for_rgb+1]

        #predict rgb using directions of ray
        feat_rgb_with_dirs=torch.cat([ feat_rgb, samples_dirs_enc],1)
        rgb=self.mlp_rgb(feat_rgb_with_dirs)


        #attemtp 2 
        density=self.softplus(density) #similar to mipnerf


        if model_colorcal is not None and img_indices is not None:
            if ray_start_end_idx is not None:
                TIME_START("colorcal")
                rgb=model_colorcal.calib_RGB_samples_packed(rgb, img_indices, ray_start_end_idx )
                TIME_END("colorcal")
            else:
                rgb=model_colorcal.calib_RGB_rays_reel(rgb, img_indices, nr_rays )

        rgb=self.sigmoid(rgb)
        

        return rgb, density

    def get_only_density(self, ray_samples, ls, iter_nr, model_colorcal=None, img_indices=None):


        # window=self.c2f(iter_nr*0.0001) #helps to converge the radiance of the object in the center and not put radiance on the walls of the bounding box
        window=self.c2f( map_range_val(iter_nr, 0.0, self.nr_iters_for_c2f, 0.3, 1.0   ) ) #helps to converge the radiance of the object in the center and not 



        #given the rays, create points
        points=ray_samples.view(-1,ray_samples.shape[-1])

      
        ray_features_full=self.encoding(points)         
        # ray_features_full=ray_features_full.float()

        #predict density without using directions
        feat_and_density=self.mlp_feat_and_density(ray_features_full)
        density=feat_and_density[:,0:1]
        density=self.softplus(density) #similar to mipnerf

        return density

    def save(self, root_folder, experiment_name, iter_nr):

        models_path=os.path.join(root_folder,"checkpoints/", experiment_name, str(iter_nr), "models")
        os.makedirs(models_path, exist_ok=True)
        if self.input_channels==3:
            torch.save(self.state_dict(), os.path.join(models_path, "ingp_model.pt")  )
        elif self.input_channels==4:
            torch.save(self.state_dict(), os.path.join(models_path, "ingp_bg_model.pt")  )

class DeferredRender(torch.nn.Module):

    def __init__(self, tex_size, tex_nr_channels, nr_textures, use_mlp, use_unet):
        super(DeferredRender, self).__init__()

        self.use_mlp=use_mlp
        self.use_unet=use_unet

        
        #make texture
        # self.texture= torch.randn( 1, tex_nr_channels, tex_size, tex_size  )*1e-4
        # self.texture=torch.nn.Parameter(self.texture)

        #make various textures
        self.nr_textures=nr_textures
        self.max_tex_size=tex_size
        self.textures = nn.ParameterList()
        for i in range(self.nr_textures):
            cur_tex_size=tex_size//pow(2,i)
            print("cur_tex_size", cur_tex_size)
            # self.textures.append( torch.nn.Parameter( torch.randn( 1, tex_nr_channels, cur_tex_size, cur_tex_size  )*1e-4  )   )
            self.textures.append( torch.nn.Parameter( torch.randn( 1, tex_nr_channels, cur_tex_size, cur_tex_size  )*0  )   )




        if self.use_mlp:
            self.mlp= nn.Sequential(
                LinearWN(tex_nr_channels, 32),
                torch.nn.Mish(),
                LinearWN(32,32),
                torch.nn.Mish(),
                LinearWN(32,32),
                torch.nn.Mish(),
                LinearWN(32,3)
            )
            apply_weight_init_fn(self, leaky_relu_init, negative_slope=0.0)
            leaky_relu_init(self.mlp[-1], negative_slope=1.0)

        if self.use_unet:
            self.unet=UNet(in_channels=tex_nr_channels, start_channels=8, nr_downsamples=2, compression_rate=1.0, out_channels=3, max_nr_channels=128)
        




        self.rgb_padding = 0.001  # Padding added to the RGB outputs.


        self.sigmoid = torch.nn.Sigmoid()



        self.c2f=Coarse2Fine(self.nr_textures)
        # self.c2f_tex=Coarse2FineTexture(iterations_to_finish=1000)
        self.c2f_tex=Coarse2FineImg(iterations_to_finish=5000)

    
    def forward(self, uv_tensor, iter_nr):

        # window=self.c2f(iter_nr*0.0003)
        # print("window", window)

        uv_tensor_channels_last=uv_tensor.permute(0,2,3,1)
        uv_tensor_channels_last_11=uv_tensor_channels_last*2-1.0
        # print("uv_tensor_channels_last", uv_tensor_channels_last.shape)

        # tex=self.texture.permute(0,2,3,1).squeeze(0)
        # if self.training:
        #     tex=self.c2f_tex(tex)
        # tex=tex.permute(2,0,1).unsqueeze(0)


        # if self.training: #backward hood cannot be applied when we are not training
            # tex=self.c2f_tex(self.texture)
        # else:
            # tex=self.texture
        # tex=self.texture

        #slice one texture
        # sampled_feat=torch.nn.functional.grid_sample(tex, uv_tensor_channels_last_11, mode='bilinear')
        # x=sampled_feat


        #slice multiple textures
        x=0
        for i in range(self.nr_textures):
            # x+= window[i]* torch.nn.functional.grid_sample(self.textures[i], uv_tensor_channels_last_11, mode='bilinear')
            x+=  torch.nn.functional.grid_sample(self.textures[i], uv_tensor_channels_last_11, mode='bilinear')

        x=x.float()

        # print("x is ", x.type())


        if self.use_unet:
            x=self.unet(x)

        # x = self.sigmoid(x)
      


        return x

    def save(self, root_folder, experiment_name, iter_nr):

        models_path=os.path.join(root_folder,"checkpoints/", experiment_name, str(iter_nr), "models")
        os.makedirs(models_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(models_path, "deferred_render_model.pt")  )



class Colorcal(torch.nn.Module):
    def __init__(self, nr_cams, idx_with_fixed_calib):
        super(Colorcal, self).__init__()


        #we need an image to have no calibration (so the weight=1 and bias=0 for it)
        #if we don't have this, the network is would be allowed to change the weight and bias arbitraraly and the RGB network would compensate and therefore would probably generalize poorly to novel views
        self.idx_with_fixed_calib=idx_with_fixed_calib 

        #instead of storing the weight which should be 1, we store the weight delta from 1. This allows us to use weight decay for this module which will keep the weight_delta close to zero 
        self.weight_delta = torch.nn.Parameter(
                torch.zeros(nr_cams, 3))
        self.bias = torch.nn.Parameter(
                torch.zeros(nr_cams, 3))

    def calib_RGB_samples_packed(self, rgb_samples, per_pixel_img_indices, ray_start_end_idx):
        # rgb_samples_contains the rgb for eahc sample in a nr_samples_total x3. This is due tot he fact that we sue an occupancy grid so eveyr ray has different number of samples
        #RGB_linear is nr_rays, nr_samples,3
        assert rgb_samples.dim()==2, "RGB_samples should have 2 dimensions corresponding to nr_samples_total, 3"
        assert rgb_samples.shape[1]==3, "RGB_samples should be nr_samples_total,3"
        #each pixel may be sampled from a different image
        assert per_pixel_img_indices.dim()==1, "per_pixel_img_indices should have 1 dimensions"


        #gather 3 weights and 3 biases for each pixel
        weights_delta_per_pixel=torch.index_select(self.weight_delta, 0, per_pixel_img_indices.long()) #nr_rays x3
        weights_per_pixel=1.0+weights_delta_per_pixel
        bias_per_pixel=torch.index_select(self.bias, 0, per_pixel_img_indices.long()) #nr_rays x3

        #for the camera that is fixed, it's weights should be 1 and bias should be 0
        fixed_pixels=per_pixel_img_indices==self.idx_with_fixed_calib
        weights_per_pixel[fixed_pixels,:]=1.0
        bias_per_pixel[fixed_pixels,:]=0.0


        #get the nr of samples per_ray
        nr_samples_per_ray=ray_start_end_idx[:,1:2]-ray_start_end_idx[:,0:1] 
        #repeat each weight and each bias, as many samples as we have for each ray
        weights_per_pixel=torch.repeat_interleave(weights_per_pixel, nr_samples_per_ray.view(-1), dim=0)
        bias_per_pixel=torch.repeat_interleave(bias_per_pixel, nr_samples_per_ray.view(-1), dim=0)

        rgb_samples=rgb_samples*weights_per_pixel+bias_per_pixel

        return rgb_samples

    def path_to_save_model(self, ckpt_folder, experiment_name, iter_nr):
        models_path=os.path.join(ckpt_folder, experiment_name, str(iter_nr), "models")
        return models_path

    def save(self, ckpt_folder, experiment_name, iter_nr):

        # models_path=os.path.join(root_folder,"checkpoints/", experiment_name, str(iter_nr), "models")
        models_path=self.path_to_save_model(ckpt_folder, experiment_name, iter_nr)
        os.makedirs(models_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(models_path, "colorcal_model.pt")  )

   