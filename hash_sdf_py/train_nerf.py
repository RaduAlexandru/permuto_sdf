#!/usr/bin/env python3

import torch

import sys
import os
import numpy as np
import argparse

from easypbr  import *
from dataloaders import *
from hash_sdf import HashSDF
from hash_sdf  import TrainParams
from hash_sdf  import NGPGui
from hash_sdf  import OccupancyGrid
from hash_sdf  import VolumeRendering
from hash_sdf  import RaySampler
from hash_sdf_py.models.models import NerfHash
from hash_sdf_py.models.models import Colorcal
from hash_sdf_py.utils.aabb import AABB
from hash_sdf_py.utils.common_utils import create_dataloader
from hash_sdf_py.utils.common_utils import create_bb_for_dataset
from hash_sdf_py.utils.common_utils import create_bb_mesh
from hash_sdf_py.utils.common_utils import show_points
from hash_sdf_py.utils.common_utils import lin2nchw
from hash_sdf_py.utils.common_utils import summary
from hash_sdf_py.utils.nerf_utils import create_samples

from hash_sdf_py.callbacks.callback_utils import *



#argparse
parser = argparse.ArgumentParser(description='Train nerf')
parser.add_argument('--dataset', default="", required=True, help='Dataset like bmvs, dtu, multiface')
parser.add_argument('--scene', default="", help='Scene name like dtu_scan24')
parser.add_argument('--comp_name', required=True,  help='Tells which computer are we using which influences the paths for finding the data')
parser.add_argument('--low_res', action='store_true', help="Use_low res images for training for when you have little GPU memory")
parser.add_argument('--with_mask', action='store_true', help="Set this to true in order to train with a mask")
parser.add_argument('--no_viewer', action='store_true', help="Set this to true in order disable the viewer")
args = parser.parse_args()
with_viewer=not args.no_viewer



config_file="train_nerf.cfg"

torch.manual_seed(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../config', config_file)


# #initialize the parameters used for training
train_params=TrainParams.create(config_path)    
class HyperParams:
    lr= 1e-3
    eikonal_weight=0.04
    lr_milestones=[100000,150000,180000,190000]
    iter_finish_training=200000
    use_occupancy_grid=True
    nr_samples_bg=32
    min_dist_between_samples=0.0001
    max_nr_samples_per_ray=64 #for the foreground
    use_color_calibration=True
    nr_rays=512
    foreground_nr_iters_for_c2f=10000
    background_nr_iters_for_c2f=10000
hyperparams=HyperParams()





def run():
    if with_viewer:
        view=Viewer.create(config_path)
        ngp_gui=NGPGui.create(view)
    

    first_time=True

    if first_time and with_viewer:
        view.m_camera.from_string(" 1.16767 0.373308  0.46992 -0.126008  0.545201 0.0833038 0.82458 -0.00165809  -0.0244027  -0.0279725 60 0.0502494 5024.94")

    experiment_name="s_"+args.scene


    loader_train, loader_test= create_dataloader(config_path, args.dataset, args.scene, args.low_res, args.comp_name, args.with_mask)

    aabb = create_bb_for_dataset(args.dataset)
    bb_mesh = create_bb_mesh(aabb) 
    Scene.show(bb_mesh,"bb_mesh")

    #make an tensorreel and get rays from all the images at
    tensor_reel=MiscDataFuncs.frames2tensors(loader_train.get_all_frames())

    cb=create_callbacks(with_viewer, train_params, experiment_name, config_path)


    #create phases
    phases= [
        Phase('train', loader_train, grad=True),
    ]
    phase=phases[0] #we usually switch between training and eval phases but here we only train 
    #model 
    model=NerfHash(3,  boundary_primitive=aabb, nr_iters_for_c2f=hyperparams.foreground_nr_iters_for_c2f ).to("cuda")
    model_bg=NerfHash(4, boundary_primitive=aabb, nr_iters_for_c2f=hyperparams.background_nr_iters_for_c2f ).to("cuda")
    # model_colorcal=Colorcal(loader_train.nr_samples(), 0)
    model_colorcal=None
    model.train(phase.grad)
    model_bg.train(phase.grad)

    params = list(model.parameters()) + list(model_bg.parameters()) 
    if model_colorcal is not None:
        params+= list(model_colorcal.parameters()) 
    optimizer = torch.optim.AdamW (params, amsgrad=False,  betas=(0.9, 0.99), eps=1e-15, lr=hyperparams.lr)


    occupancy_grid=OccupancyGrid(64, 1.0, [0,0,0])


    first_time_getting_control=True



    while ( True ): 
        model.train(phase.grad)
        model_bg.train(phase.grad)
        loss=0 

        # #forward
        cb.before_forward_pass() 

        #occupancy
        with torch.set_grad_enabled(False):
            #get random center
            if phase.iter_nr%8==0:
                grid_centers_random, grid_center_indices=occupancy_grid.compute_random_sample_of_grid_points(256*256,True)
                density_field=model.get_only_density( grid_centers_random, phase.iter_nr)  #get rgba field for all the centers
                occupancy_grid.update_with_density_random_sample(grid_center_indices,density_field, 0.7, 1e-3)  #update the occupancy
        #finish ocuppancy

        
        with torch.set_grad_enabled(False):
            ray_origins, ray_dirs, gt_selected, gt_mask, img_indices=HashSDF.random_rays_from_reel(tensor_reel, hyperparams.nr_rays) 
            ray_points_entry, ray_t_entry, ray_points_exit, ray_t_exit, does_ray_intersect_box=aabb.ray_intersection(ray_origins, ray_dirs)
            fg_ray_samples_packed, bg_ray_samples_packed = create_samples(args, hyperparams, ray_origins, ray_dirs, model.training, occupancy_grid, aabb)


        #foreground
        #compute rgb and density
        rgb_samples, density_samples=model( fg_ray_samples_packed.samples_pos, fg_ray_samples_packed.samples_dirs, phase.iter_nr, model_colorcal, img_indices, ray_start_end_idx=fg_ray_samples_packed.ray_start_end_idx) 
        #volumetric integration
        weights, weights_sum, bg_transmittance= model.volume_renderer_nerf.compute_weights(fg_ray_samples_packed, density_samples.view(-1,1))
        pred_rgb=model.volume_renderer_nerf.integrate(fg_ray_samples_packed, rgb_samples, weights)
           
                

        #run nerf bg
        if args.with_mask:
            loss_mask=torch.nn.functional.binary_cross_entropy(weights_sum.clip(1e-3, 1.0 - 1e-3), gt_mask)
            loss+=loss_mask*0.1
        else: #have to model the background
            #compute rgb and density
            rgb_samples_bg, density_samples_bg=model_bg( bg_ray_samples_packed.samples_pos_4d, bg_ray_samples_packed.samples_dirs, phase.iter_nr, model_colorcal, img_indices, ray_start_end_idx=bg_ray_samples_packed.ray_start_end_idx) 
            #volumetric integration
            weights_bg, weight_sum_bg, _= model_bg.volume_renderer_nerf.compute_weights(bg_ray_samples_packed, density_samples_bg.view(-1,1))
            pred_rgb_bg=model_bg.volume_renderer_nerf.integrate(bg_ray_samples_packed, rgb_samples_bg, weights_bg)
            #combine
            pred_rgb_bg = bg_transmittance.view(-1,1) * pred_rgb_bg
            pred_rgb = pred_rgb + pred_rgb_bg

   
            



        loss_rgb= ((gt_selected - pred_rgb)**2*does_ray_intersect_box*1.0 ).mean() 
        loss+=loss_rgb


    
        cb.after_forward_pass(loss=loss.item(), loss_rgb=loss_rgb.item(), phase=phase, lr=optimizer.param_groups[0]["lr"]) #visualizes the prediction 

        #backward
        optimizer.zero_grad()
        cb.before_backward_pass()
        loss.backward()
        # summary(model_bg)
        cb.after_backward_pass()
        optimizer.step()

            

        with torch.set_grad_enabled(False):
            model.eval()
            model_bg.eval()

            # #visualize every frame
            # if phase.iter_nr%1==0:
            #     Gui.show(frame.rgb_32f, "rgb")
            #     frustum_mesh=frame.create_frustum_mesh(0.02)
            #     Scene.show(frustum_mesh, "frustum")
            #     if (train_params.dataset_name()=="volref"):
            #         cloud=frame_depth.depth2world_xyz_mesh()
            #         frame.assign_color(cloud) #project the cloud into this frame and creates a color matrix for it
            #         Scene.show(cloud, "cloud")


            ###visualize every once in a while
            should_visualize_things=False
            if with_viewer:
                if ngp_gui.m_control_view:
                    should_visualize_things=True
            if phase.iter_nr%300==0 or phase.iter_nr==1 or should_visualize_things :


                if with_viewer:

                    # if iter_nr%100==0:
                    # show_points(ray_samples,"ray_samples")
                    if hyperparams.use_occupancy_grid:
                        show_points(fg_ray_samples_packed.samples_pos, "samples_pos")
                    # if ray_samples_bg is not None:
                    # if not args.with_mask:
                        # print("ray_samples_bg",ray_samples_bg.shape)
                        # show_points(ray_samples_bg.view(nr_rays*nr_samples_bg,-1)[:,0:3],"ray_samples_bg", color=[1.0, 0.3, 0.8])
                        # if do_importance_sampling_bg:
                            # show_points(ray_samples_bg_fine.view(-1,3),"ray_samples_bg_fine", color=[0.5, 0.3, 0.2])
                        # show_points(ray_samples_bg_4d.view(nr_rays*nr_samples_bg,-1)[:,1:4],"ray_samples_bg_4d", color=[0.3, 0.7, 0.5])
                    # show_points(pts,"pts", color=[0.3, 0.7, 0.5])




                    vis_width=150
                    vis_height=150
                    if first_time_getting_control or ngp_gui.m_control_view:
                        first_time_getting_control=False
                        frame_controlable=Frame()
                        frame_controlable.from_camera(view.m_camera, vis_width, vis_height)
                        frustum_mesh_controlable=frame_controlable.create_frustum_mesh(0.1)
                        Scene.show(frustum_mesh_controlable,"frustum_mesh_controlable")

                    #create all rays
                    ray_origins, ray_dirs=model.create_rays(frame_controlable, rand_indices=None) # ray origins and dirs as nr_pixels x 3
                    ray_points_entry, ray_t_entry, ray_points_exit, ray_t_exit, does_ray_intersect_box=aabb.ray_intersection(ray_origins, ray_dirs)
                    fg_ray_samples_packed, bg_ray_samples_packed = create_samples(args, hyperparams, ray_origins, ray_dirs, model.training, occupancy_grid, aabb)

                    # pred_rgb, ray_samples = run_nerf(model, model_bg, lattice, phase, ray_origins, ray_dirs, nr_samples_per_ray)

                    # if hyperparams.use_occupancy_grid:
                    #     #get also the ray through occupancy
                    #     ray_samples_packed=occupancy_grid.compute_samples_in_occupied_regions(ray_origins, ray_dirs, ray_t_entry, ray_t_exit, hyperparams.min_dist_between_samples, hyperparams.max_nr_samples_per_ray, model.training)
                    #     # print("initial_ray_samples_packed, cur_nr_samples is ", ray_samples_packed.cur_nr_samples)
                    #     # print("initial_ray_samples_packed, max_nr_samples is ", ray_samples_packed.max_nr_samples)
                    #     # print("initial_ray_samples_packed, samples_pos is ", ray_samples_packed.samples_pos.shape)
                    #     # print("initial_ray_samples_packed, index_max is ", ray_samples_packed.ray_start_end_idx.max())
                    #     ray_samples_packed=ray_samples_packed.get_valid_samples()
                    #     # print("after, cur_nr_samples is ", ray_samples_packed.cur_nr_samples)
                    #     # print("after, max_nr_samples is ", ray_samples_packed.max_nr_samples)
                    #     # print("after, samples_pos is ", ray_samples_packed.samples_pos.shape)




                    # #create ray samples
                    # # z_vals, dummy = model.ray_sampler.get_z_vals(ray_origins, ray_dirs, ray_t_exit, model, lattice, phase.iter_nr)
                    # # ray_samples = ray_origins.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
                    # # #repeat also the direction so that we have one direction for each sample
                    # # sample_dirs=ray_dirs.view(-1,1,3).repeat(1,nr_samples_per_ray,1).contiguous().view(-1,3)

                    # #create ray samples for bg
                    # if not args.with_mask:
                    #     # z_vals_bg, dummy, ray_samples_bg_4d, ray_samples_bg = model_bg.ray_sampler_bg.get_z_vals_bg(ray_origins, ray_dirs, model, lattice, phase.iter_nr)
                    #     # sample_dirs_bg=ray_dirs.view(-1,1,3).repeat(1,nr_samples_bg,1).contiguous().view(-1,3)
                    #     ray_samples_packed_bg= RaySampler.compute_samples_bg(ray_origins, ray_dirs, ray_t_exit, hyperparams.nr_samples_bg, aabb.m_radius, aabb.m_center_tensor, model.training, False)



                    #get rgba for every point on the ray
                    # rgb_field, density_field=model( ray_samples, ray_dirs, lattice, phase.iter_nr) 
                    # rgb_samples=rgba_field[:,:,0:3]
                    # radiance_samples=rgba_field[:,:,3]
                    # if not hyperparams.use_occupancy_grid:
                    #     rgb_field, density_field=model( ray_samples.view(-1,3), sample_dirs.view(-1,3), lattice, phase.iter_nr) 
                    #     rgb_samples=rgb_field.view(vis_height*vis_width,nr_samples_per_ray,-1)
                    #     radiance_samples=density_field.view(vis_height*vis_width,nr_samples_per_ray)
                    #     #get weights for the integration
                    #     weights, disp_map, acc_map, depth_map, bg_transmittance=model.volume_renderer(radiance_samples, z_vals, ray_t_exit)
                    #     pred_rgb = torch.sum(weights.unsqueeze(-1) * rgb_samples, 1)
                    # else:
                    #     #fused
                    #     #run the model again to get the rgba for the samples
                    #     rgb_samples, density_samples=model( ray_samples_packed.samples_pos, ray_samples_packed.samples_dirs, phase.iter_nr) 
                    #     # rgb_samples_fused=rgba_field_fused[:,0:3]
                    #     # radiance_samples_fused=rgba_field_fused[:,3]
                    #     #volrender
                    #     pred_rgb_fused, pred_depth_fused, bg_transmittance, _= model.volume_renderer_general.volume_render_nerf(ray_samples_packed, rgb_samples.view(-1,3), density_samples.view(-1,1), ray_t_exit, True)
                    #     # print("pred_rgb_fused", pred_rgb_fused)
                        
                    #     pred_rgb=pred_rgb_fused



                    #foreground
                    #compute rgb and density
                    rgb_samples, density_samples=model( fg_ray_samples_packed.samples_pos, fg_ray_samples_packed.samples_dirs, phase.iter_nr, model_colorcal, img_indices, ray_start_end_idx=fg_ray_samples_packed.ray_start_end_idx) 
                    #volumetric integration
                    weights, weights_sum, bg_transmittance= model.volume_renderer_nerf.compute_weights(fg_ray_samples_packed, density_samples.view(-1,1))
                    pred_rgb=model.volume_renderer_nerf.integrate(fg_ray_samples_packed, rgb_samples, weights)
                    
                            

                    #run nerf bg
                    if args.with_mask:
                        pass
                    else: #have to model the background
                        #compute rgb and density
                        rgb_samples_bg, density_samples_bg=model_bg( bg_ray_samples_packed.samples_pos_4d, bg_ray_samples_packed.samples_dirs, phase.iter_nr, model_colorcal, img_indices, ray_start_end_idx=bg_ray_samples_packed.ray_start_end_idx) 
                        #volumetric integration
                        weights_bg, weight_sum_bg, _= model_bg.volume_renderer_nerf.compute_weights(bg_ray_samples_packed, density_samples_bg.view(-1,1))
                        pred_rgb_bg=model_bg.volume_renderer_nerf.integrate(bg_ray_samples_packed, rgb_samples_bg, weights_bg)
                        #combine
                        pred_rgb_bg = bg_transmittance.view(-1,1) * pred_rgb_bg
                        pred_rgb = pred_rgb + pred_rgb_bg

                    #debug why is there one black pixel
                    # print("debugging the black pixel")
                    # #set to 1 the backgrund so it's for sure not black
                    # pred_rgb_fused[torch.logical_not(does_ray_intersect_box.repeat(1,3))]=1.0
                    # idx_ray_black=torch.argmin(pred_rgb_fused.sum(dim=-1))
                    # print("idx_ray_black",idx_ray_black)
                    # print("black pixel is ", pred_rgb_fused.sum(dim=-1).min())
                    # # paint it red to see it properly
                    # pred_rgb_fused[idx_ray_black,0]=1.0
                    # pred_rgb_fused[idx_ray_black,1]=0.0
                    # pred_rgb_fused[idx_ray_black,2]=0.0
                    # # print the start and and of this ray
                    # print("start end of black ray",ray_samples_packed.ray_start_end_idx[idx_ray_black,:])
                    # print("ray_samples_packed.max_nr_samples",ray_samples_packed.max_nr_samples)



                    # #run nerf bg
                    # if not args.with_mask:
                    #     rgb_samples_bg, density_samples_bg=model_bg( ray_samples_packed_bg.samples_pos_4d.view(-1,4), ray_samples_packed_bg.samples_dirs.view(-1,3), phase.iter_nr, model_colorcal, img_indices, nr_rays=hyperparams.nr_rays) 
                    #     rgb_samples_bg=rgb_samples_bg.view(vis_height*vis_width, hyperparams.nr_samples_bg, 3)
                    #     density_samples_bg=density_samples_bg.view(vis_height*vis_width, hyperparams.nr_samples_bg)
                    #     pred_rgb_bg, pred_depth_bg, _, _= model.volume_renderer_general.volume_render_nerf(ray_samples_packed_bg, rgb_samples_bg.view(-1,3), density_samples_bg.view(-1,1), ray_t_exit, False)
                    #     #combine attempt 3 like in https://github.com/lioryariv/volsdf/blob/a974c883eb70af666d8b4374e771d76930c806f3/code/model/network_bg.py#L96
                    #     pred_rgb_bg = bg_transmittance.view(-1,1) * pred_rgb_bg
                    #     pred_rgb = pred_rgb + pred_rgb_bg

                        

                    # pred_rgb_fused_img=lin2nchw(pred_rgb_fused, frame_controlable.width, frame_controlable.height)
                    # Gui.show(tensor2mat(pred_rgb_fused_img).rgb2bgr(), "pred_rgb_fused_img")
                    # pred_depth_fused_img=lin2nchw(pred_depth_fused, frame_controlable.width, frame_controlable.height)
                    # Gui.show(tensor2mat(pred_depth_fused_img), "pred_depth_fused_img")
                    # bg_transmittance_img=lin2nchw(bg_transmittance, frame_controlable.width, frame_controlable.height)
                    # Gui.show(tensor2mat(bg_transmittance_img), "bg_transmittance_img")

                    #when controlling we put them in another mehs
                    # if ngp_gui.m_control_view:
                    #     show_points(ray_samples,"ray_samples_control")
                    #     if without_mask:
                    #         show_points(ray_samples_bg.view(ray_samples.shape[0]*nr_samples_bg,-1)[:,0:3],"ray_samples_bg_control", color=[0.3, 0.7, 0.5])



                    #vis_rgb
                    pred_rgb_img=lin2nchw(pred_rgb, frame_controlable.width, frame_controlable.height)
                    Gui.show(tensor2mat(pred_rgb_img).rgb2bgr(), "pred_rgb_img")
                    #vis bg_transmittance
                    bg_transmittance_img=lin2nchw(bg_transmittance.view(-1,1), frame_controlable.width, frame_controlable.height)
                    Gui.show(tensor2mat(bg_transmittance_img), "bg_transmittance_img")

                    #vis_rgb_bg
                    if not args.with_mask:
                        pred_rgb_bg_img=lin2nchw(pred_rgb_bg, frame_controlable.width, frame_controlable.height)
                        Gui.show(tensor2mat(pred_rgb_bg_img).rgb2bgr(), "pred_rgb_bg_img_control")
                    # #vis weights 
                    # weights_sum=torch.sum(weights, 1, keepdim=True)
                    # weights_img=lin2nchw(weights_sum, frame_controlable.width, frame_controlable.height)
                    # Gui.show(tensor2mat(weights_img), "weights_img")
                    # #vis bg_transmittance
                    # bg_transmittance_img=lin2nchw(bg_transmittance.unsqueeze(1), frame_controlable.width, frame_controlable.height)
                    # Gui.show(tensor2mat(bg_transmittance_img), "bg_transmittance_img")
                    # Gui.show(tensor2mat(1.0-bg_transmittance_img), "bg_transmittance_img_inv")
                    # print("weights_img min max ", weights_img.min(), " ", weights_img.max())
                    # print("bg_transmittance_img_inv min max ", (1.0-bg_transmittance_img).min(), " ", (1.0-bg_transmittance_img).max())







        if phase.loader.is_finished():
            phase.loader.reset()


        if with_viewer:
            view.update()


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