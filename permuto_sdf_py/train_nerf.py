#!/usr/bin/env python3

import torch

import sys
import os
import numpy as np
import argparse
import math

from easypbr  import *
from dataloaders import *
import permuto_sdf 
from permuto_sdf import PermutoSDF
from permuto_sdf  import TrainParams
from permuto_sdf  import NGPGui
from permuto_sdf  import OccupancyGrid
from permuto_sdf  import VolumeRendering
from permuto_sdf  import RaySampler
from permuto_sdf_py.models.models import NerfHash
from permuto_sdf_py.models.models import Colorcal
from permuto_sdf_py.utils.aabb import AABB
from permuto_sdf_py.utils.common_utils import create_dataloader
from permuto_sdf_py.utils.common_utils import create_bb_for_dataset
from permuto_sdf_py.utils.common_utils import create_bb_mesh
from permuto_sdf_py.utils.common_utils import show_points
from permuto_sdf_py.utils.common_utils import lin2nchw
from permuto_sdf_py.utils.nerf_utils import create_samples

from permuto_sdf_py.callbacks.callback_utils import *





config_file="train_nerf.cfg"

torch.manual_seed(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../config', config_file)


# #initialize the parameters used for training
train_params=TrainParams.create(config_path)    
class HyperParams:
    lr= 1e-3
    iter_finish_training=200000
    use_occupancy_grid=True
    nr_samples_bg=32
    min_dist_between_samples=0.0001
    max_nr_samples_per_ray=64 #for the foreground
    use_color_calibration=True
    nr_rays=512
    foreground_nr_iters_for_c2f=1
    background_nr_iters_for_c2f=10000
hyperparams=HyperParams()



#do a forward pass through the model
def run_net(args, tensor_reel, hyperparams, ray_origins, ray_dirs, img_indices, model, model_bg, model_colorcal, occupancy_grid, iter_nr):
    with torch.set_grad_enabled(False):
        fg_ray_samples_packed, bg_ray_samples_packed = create_samples(args, hyperparams, ray_origins, ray_dirs, model.training, occupancy_grid, model.boundary_primitive)


    #foreground
    #compute rgb and density
    rgb_samples, density_samples=model( fg_ray_samples_packed.samples_pos, fg_ray_samples_packed.samples_dirs, iter_nr, model_colorcal, img_indices, ray_start_end_idx=fg_ray_samples_packed.ray_start_end_idx) 
    #volumetric integration
    weights, weights_sum, bg_transmittance= model.volume_renderer_nerf.compute_weights(fg_ray_samples_packed, density_samples.view(-1,1))
    pred_rgb=model.volume_renderer_nerf.integrate(fg_ray_samples_packed, rgb_samples, weights)
        
            

    #run nerf bg
    if args.with_mask:
        pred_rgb_bg=None
    else: #have to model the background
        #compute rgb and density
        rgb_samples_bg, density_samples_bg=model_bg( bg_ray_samples_packed.samples_pos_4d, bg_ray_samples_packed.samples_dirs, iter_nr, model_colorcal, img_indices, ray_start_end_idx=bg_ray_samples_packed.ray_start_end_idx) 
        #volumetric integration
        weights_bg, weight_sum_bg, _= model_bg.volume_renderer_nerf.compute_weights(bg_ray_samples_packed, density_samples_bg.view(-1,1))
        pred_rgb_bg=model_bg.volume_renderer_nerf.integrate(bg_ray_samples_packed, rgb_samples_bg, weights_bg)
        #combine
        pred_rgb_bg = bg_transmittance.view(-1,1) * pred_rgb_bg
        pred_rgb = pred_rgb + pred_rgb_bg


    return pred_rgb, pred_rgb_bg, weights_sum, fg_ray_samples_packed.samples_pos

#does forward pass through the model but breaks the rays up into chunks so that we don't run out of memory. Useful for rendering a full img
def run_net_in_chunks(frame, chunk_size, args, tensor_reel, hyperparams, model, model_bg, model_colorcal, occupancy_grid, iter_nr):
    ray_origins_full, ray_dirs_full=model.create_rays(frame, rand_indices=None)
    nr_chunks=math.ceil( ray_origins_full.shape[0]/chunk_size)
    ray_origins_list=torch.chunk(ray_origins_full, nr_chunks)
    ray_dirs_list=torch.chunk(ray_dirs_full, nr_chunks)
    pred_rgb_list=[]
    pred_rgb_bg_list=[]
    pred_weights_sum_list=[]
    pts_list=[]
    for i in range(len(ray_origins_list)):
        ray_origins=ray_origins_list[i]
        ray_dirs=ray_dirs_list[i]
        nr_rays_chunk=ray_origins.shape[0]
    
        #run net 
        pred_rgb, pred_rgb_bg, weights_sum, samples_fg=run_net(args, tensor_reel, hyperparams, ray_origins, ray_dirs, None, model, model_bg, None, occupancy_grid, iter_nr) 


        #accumulat the rgb and weights_sum
        pred_rgb_list.append(pred_rgb.detach())
        pred_rgb_bg_list.append(pred_rgb_bg.detach())
        pred_weights_sum_list.append(weights_sum.detach())
        pts_list.append(samples_fg.detach())


    #concat
    pred_rgb=torch.cat(pred_rgb_list,0)
    pred_rgb_bg=torch.cat(pred_rgb_bg_list,0)
    pred_weights_sum=torch.cat(pred_weights_sum_list,0)
    pts=torch.cat(pts_list,0)

    #reshape in imgs
    pred_rgb_img=lin2nchw(pred_rgb, frame.height, frame.width)
    pred_rgb_bg_img=lin2nchw(pred_rgb_bg, frame.height, frame.width)
    pred_weights_sum_img=lin2nchw(pred_weights_sum, frame.height, frame.width)

    return pred_rgb_img, pred_rgb_bg_img, pred_weights_sum_img, pts
   
#trains the nerf model
def train(args, config_path, hyperparams, train_params, loader_train, experiment_name, with_viewer, checkpoint_path, tensor_reel):

    if with_viewer:
        view=Viewer.create(config_path)
        ngp_gui=NGPGui.create(view)
    
    first_time=True

    if first_time and with_viewer:
        view.m_camera.from_string(" 1.16767 0.373308  0.46992 -0.126008  0.545201 0.0833038 0.82458 -0.00165809  -0.0244027  -0.0279725 60 0.0502494 5024.94")



    aabb = create_bb_for_dataset(args.dataset)
    bb_mesh = create_bb_mesh(aabb) 
    Scene.show(bb_mesh,"bb_mesh")

    cb=create_callbacks(with_viewer, train_params, experiment_name, config_path)

    #create phases
    phases= [
        Phase('train', loader_train, grad=True),
    ]
    phase=phases[0] #we usually switch between training and eval phases but here we only train 
    #model 
    model=NerfHash(3,  boundary_primitive=aabb, nr_iters_for_c2f=hyperparams.foreground_nr_iters_for_c2f ).to("cuda")
    model_bg=NerfHash(4, boundary_primitive=aabb, nr_iters_for_c2f=hyperparams.background_nr_iters_for_c2f ).to("cuda")
    if hyperparams.use_color_calibration:
        model_colorcal=Colorcal(loader_train.nr_samples(), 0)
    else:
        model_colorcal=None
    if hyperparams.use_occupancy_grid:
        occupancy_grid=OccupancyGrid(64, 1.0, [0,0,0])
    else:
        occupancy_grid=None
    model.train(phase.grad)
    model_bg.train(phase.grad)

    params = list(model.parameters()) + list(model_bg.parameters()) 
    if model_colorcal is not None:
        params+= list(model_colorcal.parameters()) 
    optimizer = torch.optim.AdamW (params, amsgrad=False,  betas=(0.9, 0.99), eps=1e-15,  weight_decay=0.0, lr=hyperparams.lr)

    first_time_getting_control=True
    is_in_training_loop=True


    while ( is_in_training_loop ): 
        model.train(phase.grad)
        model_bg.train(phase.grad)
        loss=0 

        # #forward
        cb.before_forward_pass() 

        with torch.set_grad_enabled(False):
            #update occupancy
            if phase.iter_nr%8==0:
                grid_centers_random, grid_center_indices=occupancy_grid.compute_random_sample_of_grid_points(256*256,True)
                density_field=model.get_only_density( grid_centers_random, phase.iter_nr)  #get rgba field for all the centers
                occupancy_grid.update_with_density_random_sample(grid_center_indices,density_field, 0.7, 1e-3)  #update the occupancy

            #get rays and samples 
            ray_origins, ray_dirs, gt_selected, gt_mask, img_indices=PermutoSDF.random_rays_from_reel(tensor_reel, hyperparams.nr_rays) 
            ray_points_entry, ray_t_entry, ray_points_exit, ray_t_exit, does_ray_intersect_box=aabb.ray_intersection(ray_origins, ray_dirs)
       

        #forward through the network and get the prediction
        pred_rgb, pred_rgb_bg, weights_sum, samples_pos_fg=run_net(args, tensor_reel, hyperparams, ray_origins, ray_dirs, img_indices, model, model_bg, model_colorcal, occupancy_grid, phase.iter_nr)


        #losses
        loss_rgb= ((gt_selected - pred_rgb)**2*does_ray_intersect_box*1.0 ).mean() 
        loss+=loss_rgb
        if args.with_mask:
            loss_mask=torch.nn.functional.binary_cross_entropy(weights_sum.clip(1e-3, 1.0 - 1e-3), gt_mask)
            loss+=loss_mask*0.1


    
        cb.after_forward_pass(loss=loss.item(), loss_rgb=loss_rgb.item(), phase=phase, lr=optimizer.param_groups[0]["lr"]) #visualizes the prediction 

        #backward
        optimizer.zero_grad()
        cb.before_backward_pass()
        loss.backward()
        cb.after_backward_pass()
        optimizer.step()



        #check if we finish training
        if phase.iter_nr==hyperparams.iter_finish_training+1:
            print("Finished training at iter ", phase.iter_nr)
            is_in_training_loop=False
            break 


        #save checkpoint
        if train_params.save_checkpoint() and phase.iter_nr%5000==0:
            model.save(checkpoint_path, experiment_name, phase.iter_nr)
            model_bg.save(checkpoint_path, experiment_name, phase.iter_nr, additional_name="_bg")
            if hyperparams.use_color_calibration:
                model_colorcal.save(checkpoint_path, experiment_name, phase.iter_nr)
            if hyperparams.use_occupancy_grid:
                path_to_save_model=model.path_to_save_model(checkpoint_path, experiment_name, phase.iter_nr)
                torch.save(occupancy_grid.get_grid_values(), os.path.join(path_to_save_model, "grid_values.pt")  )
                torch.save(occupancy_grid.get_grid_occupancy(), os.path.join(path_to_save_model, "grid_occupancy.pt"))



        ###visualize
        if with_viewer and phase.iter_nr%300==0 or phase.iter_nr==1 or ngp_gui.m_control_view:
            with torch.set_grad_enabled(False):
                model.eval()
                model_bg.eval()

                vis_width=150
                vis_height=150
                chunk_size=200*200
                if first_time_getting_control or ngp_gui.m_control_view:
                    first_time_getting_control=False
                    frame_controlable=Frame()
                    frame_controlable.from_camera(view.m_camera, vis_width, vis_height)
                    frustum_mesh=frame_controlable.create_frustum_mesh(0.1)
                    Scene.show(frustum_mesh,"frustum_mesh_vis")

                #run net in chunks because we might run out of memory otherwise
                pred_rgb_img, pred_rgb_bg_img, pred_weights_sum_img, pts=run_net_in_chunks(frame_controlable, chunk_size, args, tensor_reel, hyperparams, model, model_bg, model_colorcal, occupancy_grid, phase.iter_nr)

                #vis points
                show_points(samples_pos_fg,"samples_pos_fg")
                #vis_rgb
                # pred_rgb_img=lin2nchw(pred_rgb, vis_width, vis_height)
                Gui.show(tensor2mat(pred_rgb_img).rgb2bgr(), "pred_rgb_img")
                #vis_rgb_bg
                if not args.with_mask:
                    # pred_rgb_bg_img=lin2nchw(pred_rgb_bg, vis_width, vis_height)
                    Gui.show(tensor2mat(pred_rgb_bg_img).rgb2bgr(), "pred_rgb_bg_img_control")



        if with_viewer:
            view.update()




def run():
    #argparse
    parser = argparse.ArgumentParser(description='Train nerf')
    parser.add_argument('--dataset', default="", required=True, help='Dataset like bmvs, dtu, multiface')
    parser.add_argument('--scene', default="", help='Scene name like dtu_scan24')
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
    print("with_viewer", with_viewer)


    experiment_name="nerf_"+args.scene
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











    # if with_viewer:
    #     view=Viewer.create(config_path)
    #     ngp_gui=NGPGui.create(view)
    

    # first_time=True

    # if first_time and with_viewer:
    #     view.m_camera.from_string(" 1.16767 0.373308  0.46992 -0.126008  0.545201 0.0833038 0.82458 -0.00165809  -0.0244027  -0.0279725 60 0.0502494 5024.94")

    # experiment_name="s_"+args.scene


    # loader_train, loader_test= create_dataloader(config_path, args.dataset, args.scene, args.low_res, args.comp_name, args.with_mask)

    # aabb = create_bb_for_dataset(args.dataset)
    # bb_mesh = create_bb_mesh(aabb) 
    # Scene.show(bb_mesh,"bb_mesh")

    # #make an tensorreel and get rays from all the images at
    # tensor_reel=MiscDataFuncs.frames2tensors(loader_train.get_all_frames())

    # cb=create_callbacks(with_viewer, train_params, experiment_name, config_path)


    # #create phases
    # phases= [
    #     Phase('train', loader_train, grad=True),
    # ]
    # phase=phases[0] #we usually switch between training and eval phases but here we only train 
    # #model 
    # model=NerfHash(3,  boundary_primitive=aabb, nr_iters_for_c2f=hyperparams.foreground_nr_iters_for_c2f ).to("cuda")
    # model_bg=NerfHash(4, boundary_primitive=aabb, nr_iters_for_c2f=hyperparams.background_nr_iters_for_c2f ).to("cuda")
    # occupancy_grid=OccupancyGrid(64, 1.0, [0,0,0])
    # if hyperparams.use_color_calibration:
    #     model_colorcal=Colorcal(loader_train.nr_samples(), 0)
    # else:
    #     model_colorcal=None
    # model.train(phase.grad)
    # model_bg.train(phase.grad)

    # params = list(model.parameters()) + list(model_bg.parameters()) 
    # if model_colorcal is not None:
    #     params+= list(model_colorcal.parameters()) 
    # optimizer = torch.optim.AdamW (params, amsgrad=False,  betas=(0.9, 0.99), eps=1e-15, lr=hyperparams.lr)

    # first_time_getting_control=True


    # while ( True ): 
    #     model.train(phase.grad)
    #     model_bg.train(phase.grad)
    #     loss=0 

    #     # #forward
    #     cb.before_forward_pass() 

    #     with torch.set_grad_enabled(False):
    #         #update occupancy
    #         if phase.iter_nr%8==0:
    #             grid_centers_random, grid_center_indices=occupancy_grid.compute_random_sample_of_grid_points(256*256,True)
    #             density_field=model.get_only_density( grid_centers_random, phase.iter_nr)  #get rgba field for all the centers
    #             occupancy_grid.update_with_density_random_sample(grid_center_indices,density_field, 0.7, 1e-3)  #update the occupancy

    #         #get rays and samples 
    #         ray_origins, ray_dirs, gt_selected, gt_mask, img_indices=PermutoSDF.random_rays_from_reel(tensor_reel, hyperparams.nr_rays) 
    #         ray_points_entry, ray_t_entry, ray_points_exit, ray_t_exit, does_ray_intersect_box=aabb.ray_intersection(ray_origins, ray_dirs)
       

    #     #forward through the network and get the prediction
    #     pred_rgb, pred_rgb_bg, weights_sum, samples_pos_fg=run_net(args, tensor_reel, hyperparams, ray_origins, ray_dirs, img_indices, model, model_bg, model_colorcal, occupancy_grid, phase.iter_nr)


    #     #losses
    #     loss_rgb= ((gt_selected - pred_rgb)**2*does_ray_intersect_box*1.0 ).mean() 
    #     loss+=loss_rgb
    #     if args.with_mask:
    #         loss_mask=torch.nn.functional.binary_cross_entropy(weights_sum.clip(1e-3, 1.0 - 1e-3), gt_mask)
    #         loss+=loss_mask*0.1


    
    #     cb.after_forward_pass(loss=loss.item(), loss_rgb=loss_rgb.item(), phase=phase, lr=optimizer.param_groups[0]["lr"]) #visualizes the prediction 

    #     #backward
    #     optimizer.zero_grad()
    #     cb.before_backward_pass()
    #     loss.backward()
    #     cb.after_backward_pass()
    #     optimizer.step()



    #     ###visualize
    #     if with_viewer and phase.iter_nr%300==0 or phase.iter_nr==1 or ngp_gui.m_control_view:
    #         with torch.set_grad_enabled(False):
    #             model.eval()
    #             model_bg.eval()

    #             vis_width=150
    #             vis_height=150
    #             chunk_size=200*200
    #             if first_time_getting_control or ngp_gui.m_control_view:
    #                 first_time_getting_control=False
    #                 frame_controlable=Frame()
    #                 frame_controlable.from_camera(view.m_camera, vis_width, vis_height)
    #                 frustum_mesh=frame_controlable.create_frustum_mesh(0.1)
    #                 Scene.show(frustum_mesh,"frustum_mesh_vis")

    #             #run net in chunks because we might run out of memory otherwise
    #             pred_rgb_img, pred_rgb_bg_img, pred_weights_sum_img, pts=run_net_in_chunks(frame_controlable, chunk_size, args, tensor_reel, hyperparams, model, model_bg, model_colorcal, occupancy_grid, phase.iter_nr)

    #             #vis points
    #             show_points(samples_pos_fg,"samples_pos_fg")
    #             #vis_rgb
    #             # pred_rgb_img=lin2nchw(pred_rgb, vis_width, vis_height)
    #             Gui.show(tensor2mat(pred_rgb_img).rgb2bgr(), "pred_rgb_img")
    #             #vis_rgb_bg
    #             if not args.with_mask:
    #                 # pred_rgb_bg_img=lin2nchw(pred_rgb_bg, vis_width, vis_height)
    #                 Gui.show(tensor2mat(pred_rgb_bg_img).rgb2bgr(), "pred_rgb_bg_img_control")




    #     if phase.loader.is_finished():
    #         phase.loader.reset()


    #     if with_viewer:
    #         view.update()


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