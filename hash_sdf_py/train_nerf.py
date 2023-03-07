#!/usr/bin/env python3

import torch

import sys
import os
import numpy as np
import time
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
# from hash_sdf_py.hash_sdf.models import *
# from hash_sdf_py.hash_sdf.sdf_utils import *
from hash_sdf_py.utils.aabb import *
from hash_sdf_py.utils.common_utils import create_dataloader
from hash_sdf_py.utils.common_utils import create_bb_for_dataset
from hash_sdf_py.utils.common_utils import create_bb_mesh
from hash_sdf_py.utils.common_utils import show_points
from hash_sdf_py.utils.common_utils import lin2nchw

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
        view.m_camera.from_string("-1.69051 0.499783 0.824015  -0.119454    -0.5118 -0.0721315 0.847683  0.0532509 -0.0668205 -0.0933657 60 0.0502494 5024.94")

    # experiment_name="s5_permuto_mlp_siren_layer2_h32_scale30"
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
    #model 
    model=NerfHash(3,  boundary_primitive=aabb, nr_iters_for_c2f=hyperparams.foreground_nr_iters_for_c2f ).to("cuda")
    model_bg=NerfHash(4, boundary_primitive=aabb, nr_iters_for_c2f=hyperparams.background_nr_iters_for_c2f ).to("cuda")
    # model_colorcal=Colorcal(loader_train.nr_samples(), 0)
    model_colorcal=None

    occupancy_grid=OccupancyGrid(64, 1.0, [0,0,0])
    use_occupancy_grid=True


    first_time_getting_control=True

    while True:

        for phase in phases:
            cb.epoch_started(phase=phase)
            cb.phase_started(phase=phase)

            while ( True ): #we assume we have the data
                if True: 
                    is_training = phase.grad
                    model.train(phase.grad)
                    model_bg.train(phase.grad)


                    # #forward
                    with torch.set_grad_enabled(is_training):
                        cb.before_forward_pass() #sets the appropriate sigma for the lattice

                        ####New stuff 
                        with torch.set_grad_enabled(False):

                            #get random center
                            if phase.iter_nr%8==0 and use_occupancy_grid:
                                grid_centers_random, grid_center_indices=occupancy_grid.compute_random_sample_of_grid_points(256*256,True)
                                # #get rgba field for all the centers
                                density_field=model.get_only_density( grid_centers_random, phase.iter_nr) 
                                #update the occupancy
                                occupancy_grid.update_with_density_random_sample(grid_center_indices,density_field, 0.7, 1e-3)

                        ############finish new stuff


                        
                        with torch.set_grad_enabled(False):

                           
                            ray_origins, ray_dirs, gt_selected, gt_mask, img_indices=HashSDF.random_rays_from_reel(tensor_reel, hyperparams.nr_rays) 
                            ray_points_entry, ray_t_entry, ray_points_exit, ray_t_exit, does_ray_intersect_box=aabb.ray_intersection(ray_origins, ray_dirs)


                            if use_occupancy_grid:
                                ray_samples_packed=occupancy_grid.compute_samples_in_occupied_regions(ray_origins, ray_dirs, ray_t_entry, ray_t_exit, hyperparams.min_dist_between_samples, hyperparams.max_nr_samples_per_ray, model.training)
                                ray_samples_packed=ray_samples_packed.get_valid_samples()
                            #compute the nr of sampler per ray 
                            # nr_samples_per_ray=ray_samples_packed.ray_start_end_idx[:,1:2]-ray_samples_packed.ray_start_end_idx[:,0:1]
                            # print("nr_samples_per_ray",nr_samples_per_ray)
                            # print("nr_samples_per_ray",nr_samples_per_ray.shape)
                            # print("avg, min max samples_per_ray", nr_samples_per_ray.float().mean(), nr_samples_per_ray.min(), nr_samples_per_ray.max())


                            # #create ray samples
                            # TIME_START("get_z_vals") #0.6
                            # z_vals, dummy = model.ray_sampler.get_z_vals(ray_origins, ray_dirs, ray_t_exit, model, lattice, phase.iter_nr)
                            # ray_samples = ray_origins.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
                            # print("z_vals",z_vals.min(), z_vals.max())
                            # TIME_END("get_z_vals")


                            #create ray samples for bg
                            if not args.with_mask:
                                # z_vals_bg, dummy, ray_samples_bg_4d, ray_samples_bg = model_bg.ray_sampler_bg.get_z_vals_bg(ray_origins, ray_dirs, model, lattice_bg, phase.iter_nr)
                                # z_vals_bg, ray_samples_bg, ray_samples_bg_4d= RaySampler.compute_samples_bg(ray_origins, ray_dirs, ray_t_exit, nr_samples_bg, aabb.m_radius, aabb.m_center_tensor, model.training)
                                ray_samples_packed_bg= RaySampler.compute_samples_bg(ray_origins, ray_dirs, ray_t_exit, hyperparams.nr_samples_bg, aabb.m_radius, aabb.m_center_tensor, model.training, False)
                                # sample_dirs_bg=ray_dirs.view(-1,1,3).repeat(1,nr_samples_bg,1).contiguous().view(-1,3)
                                # print("ray_samples_bg_4d",ray_samples_bg_4d.shape)
                                # print("sample_dirs",sample_dirs.shape)
                                # print("ray_t_entry",ray_t_entry.shape)
                                # print("z_vals_bg",z_vals_bg.shape)
                                # print("ray_samples_bg_4d",ray_samples_bg_4d.shape)



                        #get rgba for every point on the ray
                        if not use_occupancy_grid:
                            #repeat also the direction so that we have one direction for each sample
                            sample_dirs=ray_dirs.view(-1,1,3).repeat(1,nr_samples_per_ray,1).contiguous().view(-1,3)
                            rgb_field, density_field=model( ray_samples.view(-1,3), sample_dirs.view(-1,3), lattice, phase.iter_nr, model_colorcal, img_indices, None) 
                            rgb_samples=rgb_field.view(nr_rays,nr_samples_per_ray,-1)
                            radiance_samples=density_field.view(nr_rays,nr_samples_per_ray,)
                            #get weights for the integration
                            # print("radiance_samples",radiance_samples.shape)
                            # print("z_vals",z_vals.shape)
                            weights, disp_map, acc_map, depth_map, bg_transmittance=model.volume_renderer(radiance_samples, z_vals, ray_t_exit)
                            pred_rgb = torch.sum(weights.unsqueeze(-1) * rgb_samples, 1)

                        else:
                            ####again but with fused stuff
                            rgb_samples, density_samples=model( ray_samples_packed.samples_pos, ray_samples_packed.samples_dirs, phase.iter_nr, model_colorcal, img_indices, ray_start_end_idx=ray_samples_packed.ray_start_end_idx) 
                            # rgb_samples_fused=rgba_field_fused[:,0:3]
                            # radiance_samples_fused=rgba_field_fused[:,3]
                            # TIME_START("vol_render_fused") #0.04ms
                            # pred_rgb_fused= VolumeRendering.volume_render_nerf(ray_samples_packed, rgb_samples_fused.view(-1,3), radiance_samples_fused.view(-1,1), ray_t_exit)
                            pred_rgb_fused, pred_depth_fused, bg_transmittance, weight_per_sample= model.volume_renderer_general.volume_render_nerf(ray_samples_packed, rgb_samples.view(-1,3), density_samples.view(-1,1), ray_t_exit, True)
                            # TIME_END("vol_render_fused")
                            # diff=((pred_rgb-pred_rgb_fused)**2).mean()
                            # print("diff is ",diff)
                            # saving=ray_samples_packed.samples_pos.shape[0]/ray_samples.view(-1,3).shape[0]
                            # print("computation saving ", saving)

                            # pred_rgb_fused=pred_rgb #using the color from the original rendering
                            pred_rgb=pred_rgb_fused

                        #run nerf bg
                        if not args.with_mask:
                            # rgb_samples_bg, density_samples_bg=model_bg( ray_samples_bg_4d.view(-1,4), sample_dirs_bg.view(-1,3), lattice_bg, phase.iter_nr, model_colorcal, img_indices, nr_rays=nr_rays) 
                            rgb_samples_bg, density_samples_bg=model_bg( ray_samples_packed_bg.samples_pos_4d.view(-1,4), ray_samples_packed_bg.samples_dirs.view(-1,3), phase.iter_nr, model_colorcal, img_indices, nr_rays=hyperparams.nr_rays) 
                            rgb_samples_bg=rgb_samples_bg.view(hyperparams.nr_rays, hyperparams.nr_samples_bg, 3)
                            density_samples_bg=density_samples_bg.view(hyperparams.nr_rays, hyperparams.nr_samples_bg)
                            # #get weights for the integration
                            # weights_bg, disp_map_bg, acc_map_bg, depth_map_bg, _=model_bg.volume_renderer(density_samples_bg, z_vals_bg, None)
                            pred_rgb_bg, pred_depth_bg, _, _ = model.volume_renderer_general.volume_render_nerf(ray_samples_packed_bg, rgb_samples_bg.view(-1,3), density_samples_bg.view(-1,1), ray_t_exit, False)
                            # pred_rgb_bg = torch.sum(weights_bg.unsqueeze(-1) * rgb_samples_bg, 1)
                            #combine attempt 3 like in https://github.com/lioryariv/volsdf/blob/a974c883eb70af666d8b4374e771d76930c806f3/code/model/network_bg.py#L96
                            pred_rgb_bg = bg_transmittance.view(-1,1) * pred_rgb_bg
                            pred_rgb = pred_rgb + pred_rgb_bg

                            



                        loss=0 
                        loss_rgb= ((gt_selected - pred_rgb)**2*does_ray_intersect_box*1.0 ).mean() #either way there are not BG rays outside of the bounding box so it's just better to mask that part off
                        # loss_rgb= ((gt_selected - pred_rgb_fused)**2*does_ray_intersect_box*1.0 ).mean() #either way there are not BG rays outside of the bounding box so it's just better to mask that part off
                        loss+=loss_rgb


                        # distance_to_center= ray_samples.norm(dim=-1, keepdim=False)
                        # high_distance=distance_to_center>0.4
                        # loss_density=   (radiance_samples**2)*high_distance*1.0
                        # loss_density=loss_density.mean()
                        # loss+=loss_density*0.1

                    
                       

                        # print("loss_rgb", loss_rgb)





                        #if its the first time we do a forward on the model we need to create here the optimizer because only now are all the tensors in the model instantiated
                        if first_time:
                            first_time=False
                            params=[]
                            params.append( {'params': model.parameters(), 'weight_decay': 0.0, 'lr': hyperparams.lr} )
                            if model_colorcal is not None:
                                params.append( {'params': model_colorcal.parameters(), 'weight_decay': 0.0, 'lr': hyperparams.lr } )
                            if not args.with_mask:
                                params.append( {'params': model_bg.parameters(), 'weight_decay': 0.0, 'lr': hyperparams.lr} )
                                # if do_importance_sampling_bg:
                                    # params.append( {'params': model_bg_fine.parameters(), 'weight_decay': 0.0, 'lr': lr} )

                            optimizer = torch.optim.AdamW (params, amsgrad=False,  betas=(0.9, 0.99), eps=1e-15)
                            # optimizer = RAdam(params)
                            # optimizer = apex.optimizers.FusedAdam(params, adam_w_mode=True) #1.7ms for step instead of 3.4 of the radam optimizer
                            # optimizer = apex.optimizers.FusedAdagrad(params, adagrad_w_mode=True) #1.7ms for step instead of 3.4 of the radam optimizer
                            # optimizer = torch.optim.SparseAdam (params, betas=(0.9, 0.99), eps=1e-15) #does not work for some reason
                            # optimizer = LazyAdam(params, betas=(0.9, 0.99), eps=1e-15)
                            # scheduler_lr_decay=LinearLR(optimizer, start_factor=1.0, end_factor=0.001, total_iters=30000)



                        cb.after_forward_pass(loss=loss.item(), loss_rgb=loss_rgb.item(), phase=phase, lr=optimizer.param_groups[0]["lr"], loss_sdf_surface_area=0, loss_sdf_grad=0, loss_curvature=0, loss_eikonal=0, neus_variance_mean=0) #visualizes the prediction 

                    #backward
                    if is_training:
                        optimizer.zero_grad()
                        cb.before_backward_pass()
                        loss.backward()

                        # print("-------------------------------------------------")
                        # summary(model)
                        # print("model rgb:")
                        # summary(model_rgb)


                        cb.after_backward_pass()
                        optimizer.step()

                        #test how much it would take to just add gradient on the lattice monolightic
                        # with torch.set_grad_enabled(is_training):
                        #     lv=model.lattice_values_monolithic.clone()
                        #     grad=model.lattice_values_monolithic.clone()
                        #     TIME_START("try_add")
                        #     lv=lv+grad
                        #     TIME_END("try_add")

                            

                        # scheduler_warmup.step(phase.iter_nr)
                        # scheduler_lr_decay.step()

                        # grad_clip=40
                        # torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=grad_clip, norm_type=2.0)
                        # torch.nn.utils.clip_grad_norm(parameters=model_bg.parameters(), max_norm=grad_clip, norm_type=2.0)


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

                        # print("model colorcal weight min max",model_colorcal.weight_delta.min(), model_colorcal.weight_delta.max())
                        # print("model colorcal bias min max",model_colorcal.bias.min(), model_colorcal.bias.max())

                        if with_viewer:

                            # if iter_nr%100==0:
                            # show_points(ray_samples,"ray_samples")
                            if use_occupancy_grid:
                                show_points(ray_samples_packed.samples_pos, "samples_pos")
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

                            #forward all the pixels
                            # if run_test:
                            # rand_indices=model.pick_rand_pixels(frame_controlable, nr_samples=1, pick_all_pixels=True) 
                            ray_origins, ray_dirs=model.create_rays(frame_controlable, rand_indices=None) # ray origins and dirs as nr_pixels x 3
                            ray_points_entry, ray_t_entry, ray_points_exit, ray_t_exit, does_ray_intersect_box=aabb.ray_intersection(ray_origins, ray_dirs)

                            # pred_rgb, ray_samples = run_nerf(model, model_bg, lattice, phase, ray_origins, ray_dirs, nr_samples_per_ray)

                            if use_occupancy_grid:
                                #get also the ray through occupancy
                                ray_samples_packed=occupancy_grid.compute_samples_in_occupied_regions(ray_origins, ray_dirs, ray_t_entry, ray_t_exit, hyperparams.min_dist_between_samples, hyperparams.max_nr_samples_per_ray, model.training)
                                # print("initial_ray_samples_packed, cur_nr_samples is ", ray_samples_packed.cur_nr_samples)
                                # print("initial_ray_samples_packed, max_nr_samples is ", ray_samples_packed.max_nr_samples)
                                # print("initial_ray_samples_packed, samples_pos is ", ray_samples_packed.samples_pos.shape)
                                # print("initial_ray_samples_packed, index_max is ", ray_samples_packed.ray_start_end_idx.max())
                                ray_samples_packed=ray_samples_packed.get_valid_samples()
                                # print("after, cur_nr_samples is ", ray_samples_packed.cur_nr_samples)
                                # print("after, max_nr_samples is ", ray_samples_packed.max_nr_samples)
                                # print("after, samples_pos is ", ray_samples_packed.samples_pos.shape)




                            #create ray samples
                            # z_vals, dummy = model.ray_sampler.get_z_vals(ray_origins, ray_dirs, ray_t_exit, model, lattice, phase.iter_nr)
                            # ray_samples = ray_origins.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
                            # #repeat also the direction so that we have one direction for each sample
                            # sample_dirs=ray_dirs.view(-1,1,3).repeat(1,nr_samples_per_ray,1).contiguous().view(-1,3)

                            #create ray samples for bg
                            if not args.with_mask:
                                # z_vals_bg, dummy, ray_samples_bg_4d, ray_samples_bg = model_bg.ray_sampler_bg.get_z_vals_bg(ray_origins, ray_dirs, model, lattice, phase.iter_nr)
                                # sample_dirs_bg=ray_dirs.view(-1,1,3).repeat(1,nr_samples_bg,1).contiguous().view(-1,3)
                                ray_samples_packed_bg= RaySampler.compute_samples_bg(ray_origins, ray_dirs, ray_t_exit, hyperparams.nr_samples_bg, aabb.m_radius, aabb.m_center_tensor, model.training, False)



                            #get rgba for every point on the ray
                            # rgb_field, density_field=model( ray_samples, ray_dirs, lattice, phase.iter_nr) 
                            # rgb_samples=rgba_field[:,:,0:3]
                            # radiance_samples=rgba_field[:,:,3]
                            if not use_occupancy_grid:
                                rgb_field, density_field=model( ray_samples.view(-1,3), sample_dirs.view(-1,3), lattice, phase.iter_nr) 
                                rgb_samples=rgb_field.view(vis_height*vis_width,nr_samples_per_ray,-1)
                                radiance_samples=density_field.view(vis_height*vis_width,nr_samples_per_ray)
                                #get weights for the integration
                                weights, disp_map, acc_map, depth_map, bg_transmittance=model.volume_renderer(radiance_samples, z_vals, ray_t_exit)
                                pred_rgb = torch.sum(weights.unsqueeze(-1) * rgb_samples, 1)
                            else:
                                #fused
                                #run the model again to get the rgba for the samples
                                rgb_samples, density_samples=model( ray_samples_packed.samples_pos, ray_samples_packed.samples_dirs, phase.iter_nr) 
                                # rgb_samples_fused=rgba_field_fused[:,0:3]
                                # radiance_samples_fused=rgba_field_fused[:,3]
                                #volrender
                                pred_rgb_fused, pred_depth_fused, bg_transmittance, _= model.volume_renderer_general.volume_render_nerf(ray_samples_packed, rgb_samples.view(-1,3), density_samples.view(-1,1), ray_t_exit, True)
                                # print("pred_rgb_fused", pred_rgb_fused)
                                
                                pred_rgb=pred_rgb_fused

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



                            #run nerf bg
                            if not args.with_mask:
                                rgb_samples_bg, density_samples_bg=model_bg( ray_samples_packed_bg.samples_pos_4d.view(-1,4), ray_samples_packed_bg.samples_dirs.view(-1,3), phase.iter_nr, model_colorcal, img_indices, nr_rays=hyperparams.nr_rays) 
                                rgb_samples_bg=rgb_samples_bg.view(vis_height*vis_width, hyperparams.nr_samples_bg, 3)
                                density_samples_bg=density_samples_bg.view(vis_height*vis_width, hyperparams.nr_samples_bg)
                                pred_rgb_bg, pred_depth_bg, _, _= model.volume_renderer_general.volume_render_nerf(ray_samples_packed_bg, rgb_samples_bg.view(-1,3), density_samples_bg.view(-1,1), ray_t_exit, False)
                                #combine attempt 3 like in https://github.com/lioryariv/volsdf/blob/a974c883eb70af666d8b4374e771d76930c806f3/code/model/network_bg.py#L96
                                pred_rgb_bg = bg_transmittance.view(-1,1) * pred_rgb_bg
                                pred_rgb = pred_rgb + pred_rgb_bg

                               

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
                #     cb.epoch_ended(phase=phase, model=model, save_checkpoint=train_params.save_checkpoint(), checkpoint_path=train_params.checkpoint_path() ) 
                #     cb.phase_ended(phase=phase) 
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