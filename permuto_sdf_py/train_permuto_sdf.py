#!/usr/bin/env python3

#trains PermutoSDF to recover geometry as sdf and color given only posed images
# CALL with ./permuto_sdf_py/train_permuto_sdf.py --dataset dtu --scene dtu_scan24 --comp_name comp_1 --exp_info default

import torch
import torch.nn.functional as F

import sys
import os
import numpy as np
import time
import argparse
import math
import random

import easypbr
from easypbr  import *
from dataloaders import *

import permuto_sdf
from permuto_sdf import PermutoSDF
from permuto_sdf  import TrainParams
from permuto_sdf  import NGPGui
from permuto_sdf  import OccupancyGrid
from permuto_sdf  import Sphere
from permuto_sdf  import VolumeRendering
from permuto_sdf_py.schedulers.multisteplr import MultiStepLR
from permuto_sdf_py.schedulers.warmup import GradualWarmupScheduler
from permuto_sdf_py.models.models import SDF
from permuto_sdf_py.models.models import RGB
from permuto_sdf_py.models.models import NerfHash
from permuto_sdf_py.models.models import Colorcal
from permuto_sdf_py.utils.sdf_utils import sdf_loss
from permuto_sdf_py.utils.sdf_utils import sphere_trace
from permuto_sdf_py.utils.sdf_utils import filter_unconverged_points
from permuto_sdf_py.utils.sdf_utils import importance_sampling_sdf_model
from permuto_sdf_py.utils.nerf_utils import create_rays_from_frame
from permuto_sdf_py.utils.nerf_utils import create_samples
from permuto_sdf_py.utils.common_utils import TIME_START
from permuto_sdf_py.utils.common_utils import TIME_END
from permuto_sdf_py.utils.common_utils import lin2nchw
from permuto_sdf_py.utils.common_utils import map_range_val
from permuto_sdf_py.utils.common_utils import show_points
from permuto_sdf_py.utils.common_utils import tex2img
from permuto_sdf_py.utils.common_utils import colormap
from permuto_sdf_py.utils.common_utils import create_dataloader
from permuto_sdf_py.utils.common_utils import create_bb_for_dataset
from permuto_sdf_py.utils.common_utils import create_bb_mesh
from permuto_sdf_py.utils.common_utils import summary
from permuto_sdf_py.utils.permuto_sdf_utils import get_frames_cropped
from permuto_sdf_py.utils.permuto_sdf_utils import init_losses
from permuto_sdf_py.utils.permuto_sdf_utils import get_iter_for_anneal
from permuto_sdf_py.utils.permuto_sdf_utils import loss_sphere_init
from permuto_sdf_py.utils.permuto_sdf_utils import rgb_loss
from permuto_sdf_py.utils.permuto_sdf_utils import eikonal_loss
from permuto_sdf_py.utils.permuto_sdf_utils import module_exists
from permuto_sdf_py.utils.aabb import AABB
from permuto_sdf_py.callbacks.callback_utils import *
if module_exists("apex"):
    import apex
    has_apex=True
else:
    has_apex=False
# torch.backends.cuda.matmul.allow_tf32 = True


config_file="train_permuto_sdf.cfg"

torch.manual_seed(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../config', config_file)


# #initialize the parameters used for training
train_params=TrainParams.create(config_path)    
class HyperParamsPermutoSDF:
    s_mult=1.0 #multiplier for the scheduler. Lower values mean faster convergance at the cost of some accuracy
    lr= 1e-3
    nr_iter_sphere_fit=4000*s_mult                      #nr iters for training with sphere SDF
    forced_variance_finish_iter=35000*s_mult            #nr iters until the SDF to density transform is sharp
    eikonal_weight=0.04
    curvature_weight=0.65
    lipshitz_weight=3e-6
    mask_weight=0.1
    offsurface_weight=1e-4
    iter_start_reduce_curv=50000*s_mult                 #nr iters when we reduce the curvature loss
    iter_finish_reduce_curv=iter_start_reduce_curv+1001
    lr_milestones=[100000*s_mult,150000*s_mult,180000*s_mult,190000*s_mult]
    iter_finish_training=200000*s_mult
    forced_variance_finish=0.8
    use_occupancy_grid=True
    nr_samples_bg=32
    min_dist_between_samples=0.0001
    max_nr_samples_per_ray=64 #for the foreground
    nr_samples_imp_sampling=16
    do_importance_sampling=True                         #adds nr_samples_imp_samplingx2 more samples per ray
    use_color_calibration=True
    nr_rays=512
    sdf_geom_feat_size=32
    sdf_nr_iters_for_c2f=10000*s_mult
    rgb_nr_iters_for_c2f=1
    background_nr_iters_for_c2f=1
    target_nr_of_samples=512*(64+16+16)             #the nr of rays are dynamically changed so that we use this nr of samples in a forward pass. you can reduce this for faster training or if your GPU has little VRAM
hyperparams=HyperParamsPermutoSDF()





def run_net(args, hyperparams, ray_origins, ray_dirs, img_indices, model_sdf, model_rgb, model_bg, model_colorcal, occupancy_grid, iter_nr_for_anneal,  cos_anneal_ratio, forced_variance):
    with torch.set_grad_enabled(False):
        ray_points_entry, ray_t_entry, ray_points_exit, ray_t_exit, does_ray_intersect_box=model_sdf.boundary_primitive.ray_intersection(ray_origins, ray_dirs)
        TIME_START("create_samples")
        fg_ray_samples_packed, bg_ray_samples_packed = create_samples(args, hyperparams, ray_origins, ray_dirs, model_sdf.training, occupancy_grid, model_sdf.boundary_primitive)
        
        if hyperparams.do_importance_sampling and fg_ray_samples_packed.samples_pos.shape[0]!=0:
            fg_ray_samples_packed=importance_sampling_sdf_model(model_sdf, fg_ray_samples_packed, ray_origins, ray_dirs, ray_t_exit, iter_nr_for_anneal)
        TIME_END("create_samples") #4ms in PermutoSDF

    # print("fg_ray_samples_packed.samples_pos.shape",fg_ray_samples_packed.samples_pos.shape)

    TIME_START("render_fg")    
    if fg_ray_samples_packed.samples_pos.shape[0]==0: #if we actualyl have samples for this batch fo rays
        pred_rgb=torch.zeros_like(ray_origins)
        pred_normals=torch.zeros_like(ray_origins)
        sdf_gradients=torch.zeros_like(ray_origins)
        weights_sum=torch.zeros_like(ray_origins)[:,0:1]
        bg_transmittance=torch.ones_like(ray_origins)[:,0:1]
    else:
        #foreground 
        #get sdf
        sdf, sdf_gradients, geom_feat=model_sdf.get_sdf_and_gradient(fg_ray_samples_packed.samples_pos, iter_nr_for_anneal)
        #get rgb
        rgb_samples = model_rgb( fg_ray_samples_packed.samples_pos, fg_ray_samples_packed.samples_dirs, sdf_gradients, geom_feat, iter_nr_for_anneal, model_colorcal, img_indices, fg_ray_samples_packed.ray_start_end_idx)
        #volumetric integration
        weights, weights_sum, bg_transmittance, inv_s = model_rgb.volume_renderer_neus.compute_weights(fg_ray_samples_packed, sdf, sdf_gradients, cos_anneal_ratio, forced_variance) #neus
        pred_rgb=model_rgb.volume_renderer_neus.integrate(fg_ray_samples_packed, rgb_samples, weights)

        #compute also normal by integrating the gradient
        grad_integrated_per_ray=model_rgb.volume_renderer_neus.integrate(fg_ray_samples_packed, sdf_gradients, weights)
        pred_normals=F.normalize(grad_integrated_per_ray, dim=1)
    TIME_END("render_fg") #7.2ms in PermutoSDF   



    # print("bg_ray_samples_packed.samples_pos_4d",bg_ray_samples_packed.samples_pos_4d)

    TIME_START("render_bg")    
    #run nerf bg
    if args.with_mask:
        pred_rgb_bg=None
    # else: #have to model the background
    elif bg_ray_samples_packed.samples_pos_4d.shape[0]!=0: #have to model the background
        #compute rgb and density
        rgb_samples_bg, density_samples_bg=model_bg( bg_ray_samples_packed.samples_pos_4d, bg_ray_samples_packed.samples_dirs, iter_nr_for_anneal, model_colorcal, img_indices, ray_start_end_idx=bg_ray_samples_packed.ray_start_end_idx) 
        #volumetric integration
        weights_bg, weight_sum_bg, _= model_bg.volume_renderer_nerf.compute_weights(bg_ray_samples_packed, density_samples_bg.view(-1,1))
        pred_rgb_bg=model_bg.volume_renderer_nerf.integrate(bg_ray_samples_packed, rgb_samples_bg, weights_bg)
        #combine
        pred_rgb_bg = bg_transmittance.view(-1,1) * pred_rgb_bg
        pred_rgb = pred_rgb + pred_rgb_bg
    TIME_END("render_bg")    




    # return pred_rgb, sdf_gradients, weights, weights_sum, fg_ray_samples_packed
    return pred_rgb, pred_rgb_bg, pred_normals, sdf_gradients, weights_sum, fg_ray_samples_packed

#does forward pass through the model but breaks the rays up into chunks so that we don't run out of memory. Useful for rendering a full img
def run_net_in_chunks(frame, chunk_size, args, hyperparams, model_sdf, model_rgb, model_bg, occupancy_grid, iter_nr_for_anneal, cos_anneal_ratio, forced_variance):
    ray_origins_full, ray_dirs_full=model_rgb.create_rays(frame, rand_indices=None)
    nr_chunks=math.ceil( ray_origins_full.shape[0]/chunk_size)
    ray_origins_list=torch.chunk(ray_origins_full, nr_chunks)
    ray_dirs_list=torch.chunk(ray_dirs_full, nr_chunks)
    pred_rgb_list=[]
    pred_rgb_bg_list=[]
    pred_weights_sum_list=[]
    pred_normals_list=[]
    for i in range(len(ray_origins_list)):
        ray_origins=ray_origins_list[i]
        ray_dirs=ray_dirs_list[i]
        nr_rays_chunk=ray_origins.shape[0]
    
        #run net 
        pred_rgb, pred_rgb_bg, pred_normals, sdf_gradients, weights_sum, fg_ray_samples_packed  =run_net(args, hyperparams, ray_origins, ray_dirs, None, model_sdf, model_rgb, model_bg, None, occupancy_grid, iter_nr_for_anneal, cos_anneal_ratio, forced_variance)


        #accumulat the rgb and weights_sum
        pred_rgb_list.append(pred_rgb.detach())
        pred_rgb_bg_list.append(pred_rgb_bg.detach()) if pred_rgb_bg is not None   else None
        pred_normals_list.append(pred_normals.detach())
        pred_weights_sum_list.append(weights_sum.detach())


    #concat
    pred_rgb=torch.cat(pred_rgb_list,0)
    pred_rgb_bg=torch.cat(pred_rgb_bg_list,0) if pred_rgb_bg_list else None
    pred_weights_sum=torch.cat(pred_weights_sum_list,0)
    pred_normals=torch.cat(pred_normals_list,0)

    #reshape in imgs
    pred_rgb_img=lin2nchw(pred_rgb, frame.height, frame.width)
    pred_rgb_bg_img=lin2nchw(pred_rgb_bg, frame.height, frame.width)   if pred_rgb_bg_list else None
    pred_weights_sum_img=lin2nchw(pred_weights_sum, frame.height, frame.width)
    pred_normals_img=lin2nchw(pred_normals, frame.height, frame.width)

    return pred_rgb_img, pred_rgb_bg_img, pred_normals_img, pred_weights_sum_img
   
def run_net_sphere_traced(frame, args, hyperparams, model_sdf, model_rgb, model_bg, occupancy_grid, iter_nr_for_anneal, cos_anneal_ratio, forced_variance,  nr_sphere_traces, sdf_multiplier, sdf_converged_tresh):
    ray_origins, ray_dirs=model_rgb.create_rays(frame, rand_indices=None)
  
    ray_end, ray_end_sdf, ray_end_gradient, geom_feat_end, traced_samples_packed=sphere_trace(nr_sphere_traces, ray_origins, ray_dirs, model_sdf, return_gradients=True, sdf_multiplier=sdf_multiplier, sdf_converged_tresh=sdf_converged_tresh, occupancy_grid=occupancy_grid)
    #check if we are in occupied space with the traced samples
    is_within_bounds= model_sdf.boundary_primitive.check_point_inside_primitive(ray_end)
    if hyperparams.use_occupancy_grid:
        is_in_occupied_space=occupancy_grid.check_occupancy(ray_end)
        is_within_bounds=torch.logical_and(is_in_occupied_space.view(-1), is_within_bounds.view(-1) )
    #make weigths for each sample that will be just 1 and 0 if the samples is in empty space
    weights=torch.ones_like(ray_end)[:,0:1].view(-1,1)
    weights[torch.logical_not(is_within_bounds)]=0.0 #set the samples that are outside of the occupancy grid to zero

    #we cannot use the ray_end_gradient directly because that is only defined at the samples, but now all rays may have samples because we used a occupancy grid, so we need to run the integrator
    ray_end_gradient_integrated=model_rgb.volume_renderer_neus.integrate(traced_samples_packed, ray_end_gradient, weights)
    pred_normals=F.normalize(ray_end_gradient_integrated, dim=1)
    #get also rgb
    rgb_samples = model_rgb(traced_samples_packed.samples_pos, traced_samples_packed.samples_dirs, ray_end_gradient, geom_feat_end, iter_nr_for_anneal)
    pred_rgb_integrated=model_rgb.volume_renderer_neus.integrate(traced_samples_packed, rgb_samples, weights) 
    #wegiths are per sample so they also have to be summed per ray
    pred_weights_sum, _=VolumeRendering.sum_over_each_ray(traced_samples_packed, weights)
    

    #bg we don't want for now
    pred_rgb_bg_img=None
    pred_rgb_img=lin2nchw(pred_rgb_integrated, frame.height, frame.width)
    pred_normals_img=lin2nchw(pred_normals, frame.height, frame.width)
    pred_weights_sum_img=lin2nchw(pred_weights_sum, frame.height, frame.width)



    return pred_rgb_img, pred_rgb_bg_img, pred_normals_img, pred_weights_sum_img




def train(args, config_path, hyperparams, train_params, loader_train, experiment_name, with_viewer, checkpoint_path, tensor_reel, frames_train=None, hardcoded_cam_init=True):


    #train
    if with_viewer:
        view=Viewer.create(config_path)
        ngp_gui=NGPGui.create(view)
    

    first_time=True

    if first_time and with_viewer and hardcoded_cam_init:
        view.m_camera.from_string(" 1.16767 0.373308  0.46992 -0.126008  0.545201 0.0833038 0.82458 -0.00165809  -0.0244027  -0.0279725 60 0.0502494 5024.94")
    
    aabb = create_bb_for_dataset(args.dataset)
    if with_viewer:
        bb_mesh = create_bb_mesh(aabb) 
        Scene.show(bb_mesh,"bb_mesh")

    cb=create_callbacks(with_viewer, train_params, experiment_name, config_path)


    #create phases
    phases= [
        Phase('train', loader_train, grad=True),
    ]
    phase=phases[0] #we usually switch between training and eval phases but here we only train

    #model 
    model_sdf=SDF(in_channels=3, boundary_primitive=aabb, geom_feat_size_out=hyperparams.sdf_geom_feat_size, nr_iters_for_c2f=hyperparams.sdf_nr_iters_for_c2f).to("cuda")
    model_rgb=RGB(in_channels=3, boundary_primitive=aabb, geom_feat_size_in=hyperparams.sdf_geom_feat_size, nr_iters_for_c2f=hyperparams.rgb_nr_iters_for_c2f).to("cuda")
    model_bg=NerfHash(4, boundary_primitive=aabb, nr_iters_for_c2f=hyperparams.background_nr_iters_for_c2f ).to("cuda") 
    if hyperparams.use_color_calibration:
        # model_colorcal=Colorcal(loader_train.nr_samples(), 0)
        model_colorcal=Colorcal(tensor_reel.rgb_reel.shape[0], 0)
    else:
        model_colorcal=None
    if hyperparams.use_occupancy_grid:
        occupancy_grid=OccupancyGrid(256, 1.0, [0,0,0])
    else:
        occupancy_grid=None
    model_sdf.train(phase.grad)
    model_rgb.train(phase.grad)
    model_bg.train(phase.grad)
    

    params=[]
    params.append( {'params': model_sdf.parameters(), 'weight_decay': 0.0, 'lr': hyperparams.lr, 'name': "model_sdf"} )
    params.append( {'params': model_bg.parameters(), 'weight_decay': 0.0, 'lr': hyperparams.lr, 'name': "model_bg" } )
    params.append( {'params': model_rgb.parameters_only_encoding(), 'weight_decay': 0.0, 'lr': hyperparams.lr, 'name': "model_rgb_only_encoding"} )
    params.append( {'params': model_rgb.parameters_all_without_encoding(), 'weight_decay': 0.0, 'lr': hyperparams.lr, 'name': "model_rgb_all_without_encoding"} )
    if model_colorcal is not None:
        params.append( {'params': model_colorcal.parameters(), 'weight_decay': 1e-1, 'lr': hyperparams.lr, 'name': "model_colorcal" } )
    if has_apex:
        optimizer = apex.optimizers.FusedAdam (params, amsgrad=False,  betas=(0.9, 0.99), eps=1e-15, weight_decay=0.0, lr=hyperparams.lr)
    else:
        optimizer = torch.optim.AdamW (params, amsgrad=False,  betas=(0.9, 0.99), eps=1e-15, weight_decay=0.0, lr=hyperparams.lr)
    scheduler_lr_decay= MultiStepLR(optimizer, milestones=hyperparams.lr_milestones, gamma=0.3, verbose=False)    


    first_time_getting_control=True
    is_in_training_loop=True
    nr_rays_to_create=hyperparams.nr_rays
   
    while is_in_training_loop:
        model_sdf.train(phase.grad)
        model_rgb.train(phase.grad)
        model_bg.train(phase.grad)
        loss=0 

        TIME_START("fw_back")

        cb.before_forward_pass()

        loss, loss_rgb, loss_eikonal, loss_curvature, loss_lipshitz=init_losses() 

        iter_nr_for_anneal=get_iter_for_anneal(phases[0].iter_nr, hyperparams.nr_iter_sphere_fit)
        in_process_of_sphere_init=phases[0].iter_nr<hyperparams.nr_iter_sphere_fit
        just_finished_sphere_fit=phases[0].iter_nr==hyperparams.nr_iter_sphere_fit

        if in_process_of_sphere_init:
            loss, loss_sdf, loss_eikonal= loss_sphere_init(args.dataset, 30000, aabb, model_sdf, iter_nr_for_anneal )
            cos_anneal_ratio=1.0
            forced_variance=0.8
        else:
            with torch.set_grad_enabled(False):
                cos_anneal_ratio=map_range_val(iter_nr_for_anneal, 0.0, hyperparams.forced_variance_finish_iter, 0.0, 1.0)
                forced_variance=map_range_val(iter_nr_for_anneal, 0.0, hyperparams.forced_variance_finish_iter, 0.3, hyperparams.forced_variance_finish)

                ray_origins, ray_dirs, gt_selected, gt_mask, img_indices=PermutoSDF.random_rays_from_reel(tensor_reel, nr_rays_to_create) 
                ray_points_entry, ray_t_entry, ray_points_exit, ray_t_exit, does_ray_intersect_box=aabb.ray_intersection(ray_origins, ray_dirs)



            TIME_START("run_net")
            pred_rgb, pred_rgb_bg, pred_normals, sdf_gradients, weights_sum, fg_ray_samples_packed  =run_net(args, hyperparams, ray_origins, ray_dirs, img_indices, model_sdf, model_rgb, model_bg, model_colorcal, occupancy_grid, iter_nr_for_anneal,  cos_anneal_ratio, forced_variance)
            TIME_END("run_net")
            

            
            #losses -----
            #rgb loss
            loss_rgb=rgb_loss(gt_selected, pred_rgb, does_ray_intersect_box)
            loss+=loss_rgb

            #eikonal loss
            loss_eikonal =eikonal_loss(sdf_gradients)
            loss+=loss_eikonal*hyperparams.eikonal_weight


            #curvature loss
            global_weight_curvature=map_range_val(iter_nr_for_anneal, hyperparams.iter_start_reduce_curv, hyperparams.iter_finish_reduce_curv, 1.0, 0.000) #once we are converged onto good geometry we can safely descrease it's weight so we learn also high frequency detail geometry.
            if global_weight_curvature>0.0:
            # if True:
                sdf_shifted, sdf_curvature=model_sdf.get_sdf_and_curvature_1d_precomputed_gradient_normal_based( fg_ray_samples_packed.samples_pos, sdf_gradients, iter_nr_for_anneal)
                loss_curvature=sdf_curvature.mean() 
                loss+=loss_curvature* hyperparams.curvature_weight*global_weight_curvature



            #loss for empty space sdf            
            if hyperparams.use_occupancy_grid:
                #highsdf just to avoice voxels becoming "occcupied" due to their sdf dropping to zero
                offsurface_points=model_sdf.boundary_primitive.rand_points_inside(nr_points=1024)
                sdf_rand, _=model_sdf( offsurface_points, iter_nr_for_anneal)
                loss_offsurface_high_sdf=torch.exp(-1e2 * torch.abs(sdf_rand)).mean()
                loss+=loss_offsurface_high_sdf*hyperparams.offsurface_weight

            #loss on lipshitz
            loss_lipshitz=model_rgb.mlp.lipshitz_bound_full()
            if iter_nr_for_anneal>=hyperparams.iter_start_reduce_curv:
                loss+=loss_lipshitz.mean()*hyperparams.lipshitz_weight

            #loss mask
            if args.with_mask:
                loss_mask=torch.nn.functional.binary_cross_entropy(weights_sum.clip(1e-3, 1.0 - 1e-3), gt_mask)
                loss+=loss_mask*hyperparams.mask_weight


            with torch.set_grad_enabled(False):
                #update occupancy
                if phase.iter_nr%8==0 and hyperparams.use_occupancy_grid:
                    grid_centers_random, grid_center_indices=occupancy_grid.compute_random_sample_of_grid_points(256*256*4,True)
                    sdf_grid,_=model_sdf( grid_centers_random, iter_nr_for_anneal) 
                    occupancy_grid.update_with_sdf_random_sample(grid_center_indices, sdf_grid, model_rgb.volume_renderer_neus.get_last_inv_s(), 1e-4 )
                    # occupancy_grid.update_with_sdf_random_sample(grid_center_indices, sdf_grid, model_rgb.volume_renderer_neus.get_last_inv_s().item(), 1e-4 )

                #adjust nr_rays_to_create based on how many samples we have in total
                cur_nr_samples=fg_ray_samples_packed.samples_pos.shape[0]
                multiplier_nr_samples=float(hyperparams.target_nr_of_samples)/cur_nr_samples
                nr_rays_to_create=int(nr_rays_to_create*multiplier_nr_samples)

                #increase also the WD on the encoding of the model_rgb to encourage the network to get high detail using the model_sdf
                if iter_nr_for_anneal>=hyperparams.iter_start_reduce_curv:
                    for group in optimizer.param_groups:
                        if group["name"]=="model_rgb_only_encoding":
                            group["weight_decay"]=1.0
                        #decrease eik_w as it seems to also slightly help with getting more detail on the surface
                        hyperparams.eikonal_weight=0.01

        # cb.after_forward_pass(loss=loss.item(), loss_rgb=loss_rgb, loss_sdf_surface_area=0, loss_sdf_grad=0, phase=phase, loss_eikonal=loss_eikonal.item(), loss_curvature=loss_curvature.item(), loss_lipshitz=loss_lipshitz.item(), lr=optimizer.param_groups[0]["lr"]) #visualizes the prediction 
        cb.after_forward_pass(loss=loss, loss_rgb=loss_rgb, phase=phase, loss_eikonal=loss_eikonal, loss_curvature=loss_curvature, loss_lipshitz=loss_lipshitz, lr=optimizer.param_groups[0]["lr"]) #visualizes the prediction 


        #backward
        optimizer.zero_grad()
        cb.before_backward_pass()
        TIME_START("backward")
        loss.backward()
        TIME_END("backward") 
        cb.after_backward_pass()
        optimizer.step()
        if just_finished_sphere_fit:
            scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=3000, after_scheduler=scheduler_lr_decay) 
        if not in_process_of_sphere_init:
            scheduler_warmup.step() #this will call the scheduler for the decay
        if phase.iter_nr==hyperparams.iter_finish_training+1:
            print("Finished training at iter ", phase.iter_nr)
            is_in_training_loop=False
            break 


        TIME_END("fw_back") #takes 56ms in ingp2, 62ms in PermutoSDF



        #print every once in a while 
        if phase.iter_nr%1000==0:
            print("phase.iter_nr",  phase.iter_nr, "loss ", loss.item() )




        if with_viewer:
            view.update()

        #save checkpoint
        if train_params.save_checkpoint() and phase.iter_nr%10000==0:
            model_sdf.save(checkpoint_path, experiment_name, phase.iter_nr)
            model_rgb.save(checkpoint_path, experiment_name, phase.iter_nr)
            model_bg.save(checkpoint_path, experiment_name, phase.iter_nr, additional_name="_bg")
            if hyperparams.use_color_calibration:
                model_colorcal.save(checkpoint_path, experiment_name, phase.iter_nr)
            if hyperparams.use_occupancy_grid:
                path_to_save_model=model_sdf.path_to_save_model(checkpoint_path, experiment_name, phase.iter_nr)
                torch.save(occupancy_grid.get_grid_values(), os.path.join(path_to_save_model, "grid_values.pt")  )
                torch.save(occupancy_grid.get_grid_occupancy(), os.path.join(path_to_save_model, "grid_occupancy.pt"))


        ###visualize
        if with_viewer and (phase.iter_nr%300==0 or phase.iter_nr==1 or ngp_gui.m_control_view):
            with torch.set_grad_enabled(False):
                model_sdf.eval()
                model_rgb.eval()
                model_bg.eval()

                if not in_process_of_sphere_init:
                    show_points(fg_ray_samples_packed.samples_pos,"samples_pos_fg")

                vis_width=300
                vis_height=300
                if first_time_getting_control or ngp_gui.m_control_view:
                    first_time_getting_control=False
                    frame=Frame()
                    frame.from_camera(view.m_camera, vis_width, vis_height)
                    frustum_mesh=frame.create_frustum_mesh(0.1)
                    Scene.show(frustum_mesh,"frustum_mesh_vis")

                #forward all the pixels
                ray_origins, ray_dirs=create_rays_from_frame(frame, rand_indices=None) # ray origins and dirs as nr_pixels x 3
                

                #render image either volumetrically or with sphere tracing
                use_volumetric_render=False
                if use_volumetric_render:
                    chunk_size=50*50
                    pred_rgb_img, pred_rgb_bg_img, pred_normals_img, pred_weights_sum_img=run_net_in_chunks(frame, chunk_size, args, hyperparams, model_sdf, model_rgb, model_bg, occupancy_grid, iter_nr_for_anneal, cos_anneal_ratio, forced_variance)
                else:
                    pred_rgb_img, pred_rgb_bg_img, pred_normals_img, pred_weights_sum_img=run_net_sphere_traced(frame, args, hyperparams, model_sdf, model_rgb, model_bg, occupancy_grid, iter_nr_for_anneal, cos_anneal_ratio, forced_variance,  nr_sphere_traces=15, sdf_multiplier=0.9, sdf_converged_tresh=0.0002)
                #vis normals
                pred_normals_img_vis=(pred_normals_img+1.0)*0.5
                pred_normals_img_vis_alpha=torch.cat([pred_normals_img_vis,pred_weights_sum_img],1)
                Gui.show(tensor2mat(pred_normals_img_vis_alpha).rgba2bgra(), "pred_normals_img_vis")
                #vis RGB
                Gui.show(tensor2mat(pred_rgb_img).rgb2bgr(), "pred_rgb_img")



        #view also in tensorboard some imags
        if (phase.iter_nr%5000==0 or phase.iter_nr==1 or just_finished_sphere_fit) and train_params.with_tensorboard() and not in_process_of_sphere_init:
            with torch.set_grad_enabled(False):
                model_sdf.eval()
                model_rgb.eval()
                model_bg.eval()

                # if isinstance(loader_train, DataLoaderPhenorobCP1):
                    # frame=random.choice(frames_train)
                # else:
                if frames_train is not None:
                    frame=random.choice(frames_train)
                else:
                    frame=phase.loader.get_random_frame() #we just get this frame so that the tensorboard can render from this frame

                #make from the gt frame a smaller frame until we reach a certain size
                frame_subsampled=frame.subsample(2.0, subsample_imgs=False)
                while min(frame_subsampled.width, frame_subsampled.height) >400:
                    frame_subsampled=frame_subsampled.subsample(2.0, subsample_imgs=False)
                vis_width=frame_subsampled.width
                vis_height=frame_subsampled.height
                frame=frame_subsampled


                chunk_size=1000
                pred_rgb_img, pred_rgb_bg_img, pred_normals_img, pred_weights_sum_img=run_net_in_chunks(frame, chunk_size, args, hyperparams, model_sdf, model_rgb, model_bg, occupancy_grid, iter_nr_for_anneal, cos_anneal_ratio, forced_variance)
                # pred_rgb_img, pred_rgb_bg_img, pred_normals_img, pred_weights_sum_img=run_net_sphere_traced(frame, args, hyperparams, model_sdf, model_rgb, model_bg, occupancy_grid, iter_nr_for_anneal, cos_anneal_ratio, forced_variance,  nr_sphere_traces=15, sdf_multiplier=0.9, sdf_converged_tresh=0.0002)
                #vis normals
                pred_normals_img_vis=(pred_normals_img+1.0)*0.5
                pred_normals_img_vis_alpha=torch.cat([pred_normals_img_vis,pred_weights_sum_img],1)

                cb["tensorboard_callback"].tensorboard_writer.add_image('permuto_sdf/' + phase.name + '/pred_rgb_img', pred_rgb_img.squeeze(), phase.iter_nr)
                cb["tensorboard_callback"].tensorboard_writer.add_image('permuto_sdf/' + phase.name + '/pred_normals', pred_normals_img_vis_alpha.squeeze(), phase.iter_nr)


        # if with_viewer:
        #     view.update()
      

                   


                  


    print("finished trainng")
    return




def run():

    #argparse
    parser = argparse.ArgumentParser(description='Train sdf and color')
    parser.add_argument('--dataset', required=True, help='Dataset like bmvs, dtu, multiface')
    parser.add_argument('--scene', required=True, help='Scene name like dtu_scan24')
    parser.add_argument('--comp_name', required=True,  help='Tells which computer are we using which influences the paths for finding the data')
    parser.add_argument('--low_res', action='store_true', help="Use_low res images for training for when you have little GPU memory")
    parser.add_argument('--exp_info', default="", help='Experiment info string useful for distinguishing one experiment for another')
    parser.add_argument('--with_mask', action='store_true', help="Set this to true in order to train with a mask")
    parser.add_argument('--no_viewer', action='store_true', help="Set this to true in order disable the viewer")
    args = parser.parse_args()
    with_viewer=not args.no_viewer

    #get the checkpoints path which will be at the root of the permuto_sdf package 
    permuto_sdf_root=os.path.dirname(os.path.abspath(permuto_sdf.__file__))
    checkpoint_path=os.path.join(permuto_sdf_root, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)


    print("args.with_mask", args.with_mask)
    print("args.low_res", args.low_res)
    print("checkpoint_path",checkpoint_path)
    print("with_viewer", with_viewer)
    print("has_apex", has_apex)


    experiment_name="permuto_sdf_"+args.scene
    if args.exp_info:
        experiment_name+="_"+args.exp_info


    loader_train, loader_test= create_dataloader(config_path, args.dataset, args.scene, args.low_res, args.comp_name, args.with_mask)

    #tensoreel
    if isinstance(loader_train, DataLoaderPhenorobCP1):
        aabb = create_bb_for_dataset(args.dataset)
        tensor_reel=MiscDataFuncs.frames2tensors( get_frames_cropped(loader_train, aabb) ) #make an tensorreel and get rays from all the images at
    else:
        tensor_reel=MiscDataFuncs.frames2tensors(loader_train.get_all_frames()) #make an tensorreel and get rays from all the images at



    train(args, config_path, hyperparams, train_params, loader_train, experiment_name, with_viewer, checkpoint_path, tensor_reel)

    #finished training
    return


  


def main():
    run()



if __name__ == "__main__":
     main()  # This is what you would have, but the following is useful:

    # # These are temporary, for debugging, so meh for programming style.
    # import sys, trace

    # # If there are segfaults, it's a good idea to always use stderr as it
    # # always prints to the screen, so you should get as much output as
    # # possible.
    # sys.stdout = sys.stderr

    # # Now trace execution:
    # tracer = trace.Trace(trace=1, count=0, ignoredirs=["/usr", sys.prefix])
    # tracer.run('main()')
