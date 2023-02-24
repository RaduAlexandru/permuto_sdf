#!/usr/bin/env python3

import torch
import torch.nn.functional as F

import sys
import os
import numpy as np
import time

from easypbr  import *
from hash_sdf  import TrainParams
# from hash_sdf  import ModelParams
# from hash_sdf  import EvalParams
from hash_sdf  import NGPGui
from hash_sdf  import OccupancyGrid
from hash_sdf  import Sphere
from hash_sdf_py.models.models import *
from hash_sdf_py.utils.sdf_utils import sdf_loss
from hash_sdf_py.utils.sdf_utils import sphere_trace
from hash_sdf_py.utils.sdf_utils import filter_unconverged_points
from hash_sdf_py.utils.nerf_utils import create_rays_from_frame
from hash_sdf_py.utils.common_utils import show_points
from hash_sdf_py.utils.common_utils import tex2img
from hash_sdf_py.utils.common_utils import colormap
from hash_sdf_py.utils.aabb import AABB
# from hash_sdf_py.utils.sphere import *

from hash_sdf_py.callbacks.callback import *
from hash_sdf_py.callbacks.viewer_callback import *
from hash_sdf_py.callbacks.visdom_callback import *
from hash_sdf_py.callbacks.tensorboard_callback import *
from hash_sdf_py.callbacks.state_callback import *
from hash_sdf_py.callbacks.phase import *
# from hash_sdf_py.utils.utils import tex2img
# from hash_sdf_py.utils.utils import colormap
# from hash_sdf_py.utils.utils import summary

from hash_sdf_py.optimizers.radam import *
# from hash_sdf_py.optimizers.adan import *
# from hash_sdf_py.optimizers.vectoradam import *
# from hash_sdf_py.optimizers.lazyadam import *
# from hash_sdf_py.optimizers.shampoo import *
# from hash_sdf_py.schedulers.linearlr import *
# from hash_sdf_py.optimizers.grad_scaler import *

# import torch_optimizer as optim

import apex


config_file="train_sdf_from_mesh.cfg"

torch.manual_seed(0)
# torch.set_printoptions(threshold=27)
# torch.set_printoptions(profile="full")
torch.set_printoptions(edgeitems=27)
# torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type(torch.cuda.FloatTensor)
config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../config', config_file)


# #initialize the parameters used for training
train_params=TrainParams.create(config_path)    
# model_params=ModelParams.create(config_path)    



# def color_by_idx(nr_voxels):
#     #color by idx
#     colors_t=torch.range(0,nr_voxels-1)
#     #repeat and assign color
#     colors_t=colors_t.view(-1,1).repeat(1,3)
#     colors_t=colors_t.float()/nr_voxels

#     return colors_t

# def color_by_density_from_occupancy_grid(occupancy_grid):
#     #color by grid value
#     colors_t=occupancy_grid.get_grid_values()
#     colors_t=torch.clamp(colors_t,max=1.0)
#     #repeat and assign color
#     colors_t=colors_t.view(-1,1).repeat(1,3)
#     colors_t=colors_t.float()

#     return colors_t

# def color_by_occupancy_from_occupancy_grid(occupancy_grid):
#     #color by grid value
#     colors_t=occupancy_grid.get_grid_occupancy().float()
#     colors_t=torch.clamp(colors_t,max=1.0)
#     #repeat and assign color
#     colors_t=colors_t.view(-1,1).repeat(1,3)
#     colors_t=colors_t.float()

#     return colors_t

# def color_by_density(density):
#     #color by grid value
#     colors_t=density
#     colors_t=torch.clamp(colors_t,max=1.0)
#     #repeat and assign color
#     colors_t=colors_t.view(-1,1).repeat(1,3)
#     colors_t=colors_t.float()

#     return colors_t



def run():
    if train_params.with_viewer():
        view=Viewer.create(config_path)
        ngp_gui=NGPGui.create(view)
    

    first_time=True

    if first_time:
        view.m_camera.from_string("-0.551362  0.211048  0.212642  -0.126726  -0.512556 -0.0768056 0.845737   0.0638516 -0.00169279   -0.108451 60 0.0502494 5024.94")
        # view.m_camera.from_string(" -1.9268 -1.45101 -2.35208 -0.0830739   0.914755  -0.244731 -0.310516  -0.143801   0.236998 -0.0284275 60 0.0502535 5025.35")

    # experiment_name="s5_permuto_mlp_siren_layer2_h32_scale30"
    # experiment_name="sdf14_lvl24_concatpoints_enc4_1em3"
    # experiment_name="sdf17_lvl24_concatpoints_enc0_1em3_modsin_fixed_lr1e5"

    experiment_name="sdf_def"
    # experiment_name="sdf_ingp_grid"

    #params
    nr_lattice_features=2
    nr_resolutions=24
    # nr_lattice_features=1
    # nr_resolutions=64


    #torch stuff 
    # lattice=Lattice.create(config_path, "lattice")
    # nr_lattice_vertices=lattice.capacity()
    # print("nr_lattice_vertices ", nr_lattice_vertices)

    #create bounding box for the scene 
    # aabb=Sphere(0.5, [0,0,0])
    aabb=AABB(bounding_box_sizes_xyz=[1.0, 1.0, 1.0], bounding_box_translation=[0.0, 0.0, 0.0])
    

    # #create a bounding primitive
    # sphere=Sphere(radius=0.5*np.sqrt(3), center=[0.0, 0.0, 0.0])
    # sphere_mesh=Mesh()
    # sphere_mesh.create_sphere(sphere.sphere_center, sphere.sphere_radius)
    # sphere_mesh.translate_model_matrix(sphere.sphere_center)
    # sphere_mesh.m_vis.m_show_mesh=False
    # sphere_mesh.m_vis.m_show_wireframe=True
    # Scene.show(sphere_mesh,"sphere_mesh")


    #get the mesh for which we will compute the sdf
    # mesh=Mesh("/media/rosu/Data/data/phenorob/days_on_field/2021_06_02_just_15/15_processed/block_0/colmap_data/cloud_dense.ply")
    mesh=Mesh("/media/rosu/Data/data/3d_objs/artec_3d/statue-dragonfly-tamer-stl/Statue Dragonfly tamer_stl.stl")
    # mesh=Mesh("/media/rosu/Data/data/3d_objs/my_mini_factory/babygroot-pot-03-fixed.stl")
    # mesh=Mesh("/media/rosu/Data/data/3d_objs/scan_the_world/smk16-kas115-pieta-michelangelo.stl")
    # mesh=Mesh("/media/rosu/Data/data/3d_objs/artec_3d/flower-obj_0/Flower.obj/Flower.obj")
    # mesh=Mesh("/media/rosu/Data/data/3d_objs/artec_3d/dragon_and_phoenix_statuette-stl/Dragon and phoenix statuette.stl/Dragon and phoenix statuette.stl")
    # mesh=Mesh("/media/rosu/Data/data/3d_objs/artec_3d/motorcycle-wheel-ply/Motorcycle wheel.ply")
    # mesh=Mesh("/media/rosu/Data/data/3d_objs/artec_3d/damaliscus_korrigum_ply/Damaliscus Korrigum PLY/damaliscus_korrigum.ply")
    # mesh=Mesh("/media/rosu/Data/data/3d_objs/plants/corn/10439_Corn_Field_v1_L3.123c80b91965-ab50-4dd0-8508-3847dcd0c84e/10439_Corn_Field_v1_max2010_it2.obj")
    # mesh.upsample(3,True)
    mesh.model_matrix.rotate_axis_angle([1,0,0],-90)
    mesh.apply_model_matrix_to_cpu(True)
    mesh.normalize_size()
    mesh.normalize_position()
    mesh.recalculate_normals()
    # mesh.fix_oversplit_due_to_blender_uv()
    # mesh.upsample(3,True)

    # #rescale so that some parts are outside the box
    # mesh.scale_mesh(8.0)
    # mesh.model_matrix.translate([0.2, 0.5, 0])
    # mesh.apply_model_matrix_to_cpu(True)
    # #remove points outside of box
    # mesh=remove_points_outside_of_box(mesh,bounding_box_sizes_xyz, bounding_box_translation)

    Scene.show(mesh, "mesh")
    #prepare point and normal
    gt_points=torch.from_numpy(mesh.V.copy()).cuda().float()
    gt_normals=torch.from_numpy(mesh.NV.copy()).cuda().float()


    # mesh=Mesh()     #create an empty mesh
    # mesh.V=[        #fill up the vertices of the mesh as a matrix of Nx3
    #     [0,0,0],
    #     [1e-5,0,0],
    # ]
    # mesh.m_vis.m_show_points=True
    # Scene.show(mesh,"mesh_points")



    cb_list = []
    if(train_params.with_visdom()):
        cb_list.append(VisdomCallback(experiment_name))
    if(train_params.with_tensorboard()):
        cb_list.append(TensorboardCallback(experiment_name))
    if(train_params.with_viewer()):
        cb_list.append(ViewerCallback())
    cb_list.append(StateCallback())
    cb = CallbacksGroup(cb_list)


    #create loaders
    # loader_train=create_loader(train_params.dataset_name(), config_path)
    # loader_train.start()
    #create phases
    phases= [
        Phase('train', None, grad=True),
    ]
    #model 
    model=SDF(boundary_primitive=aabb, geom_feat_size_out=0, nr_iters_for_c2f=1000).to("cuda")

    first_time_getting_control=True
    run_test=True

    # sphere=Mesh()
    # sphere.create_sphere([0,0,0],0.03)
    # Scene.show(sphere,"sphere")

    mesh_point=Mesh()
    mesh_point.V=[        
        [0,0,0]
    ]
    mesh_point.C=[        
        [0,1,0]
    ]
    mesh_point.m_vis.m_show_points=True
    mesh_point.m_vis.set_color_pervertcolor()
    Scene.show(mesh_point,"mesh_point")


    occupancy_grid=OccupancyGrid(64, 1.0, [0,0,0])

    #debug
    box=Mesh()
    dist=0.001
    box.create_box(dist, dist, dist)
    box.m_vis.m_show_mesh=False
    box.m_vis.m_show_wireframe=True
    Scene.show(box,"box")

    while True:

        for phase in phases:
            cb.epoch_started(phase=phase)
            cb.phase_started(phase=phase)

            while ( True ): #we assume we have the data
                if True:
                    is_training = phase.grad
                    model.train(phase.grad)


                    # #forward
                    # TIME_START("FW_BW")
                    with torch.set_grad_enabled(is_training):
                        cb.before_forward_pass() #sets the appropriate sigma for the lattice


                        use_occupancy_grid=False
                        if phase.iter_nr%8==0 and use_occupancy_grid:
                            with torch.set_grad_enabled(False):
                                # TIME_START("update_grid")
                                grid_centers_random, grid_center_indices=occupancy_grid.compute_random_sample_of_grid_points(256*256,True)
                                # TIME_END("get_centers_rand")
                                # show_points(grid_centers_random,"grid_centers_random")

                                # print('grid_centers',grid_centers)


                                # #get rgba field for all the centers
                                # density_field=model.get_only_density( grid_centers, lattice, phase.iter_nr) 
                                sdf,_,_=model( grid_centers_random, lattice, phase.iter_nr, False) 

                                # # print("density_field",density_field)

                                #show grid centers
                                grid_centers=occupancy_grid.compute_grid_points(False)
                                grid_centers_eig=tensor2eigen(grid_centers)
                                mesh_centers=Mesh()
                                mesh_centers.V=grid_centers_eig
                                # mesh_centers.C=tensor2eigen(color_by_idx(occupancy_grid.get_nr_voxels()))
                                # mesh_centers.C=tensor2eigen(color_by_density_from_occupancy_grid(occupancy_grid))
                                mesh_centers.C=tensor2eigen(color_by_occupancy_from_occupancy_grid(occupancy_grid))
                                # mesh_centers.C=tensor2eigen(color_by_density(density_field))
                                mesh_centers.m_vis.m_show_points=True
                                mesh_centers.m_vis.set_color_pervertcolor()
                                Scene.show(mesh_centers,"mesh_centers")

                                print("sdf min max is ", sdf.min(), sdf.max())

                                #update the occupancy
                                # occupancy_grid.update_with_density(density_field, 0.95, 1e-3)
                                inv_s=math.exp(0.7*10)
                                max_eikonal_abs=0.0
                                occupancy_thresh=1e-6
                                occupancy_grid.update_with_sdf_random_sample(grid_center_indices, sdf, inv_s, max_eikonal_abs, occupancy_thresh )
                                # TIME_END("update_grid")


                        #sample some random point on the surface 
                        multiplier=1
                        # rand_indices=model.pick_rand_rows(gt_points, nr_samples=3000*multiplier, pick_all=False) #returns None if we select all the rows
                        # surface_points=model.row_sampler(gt_points, rand_indices) #returns the full GT if rand_indices is None. Returns Nx3
                        # surface_normals=model.row_sampler(gt_normals, rand_indices) #returns the full GT if rand_indices is None. Returns Nx3
                        rand_indices=torch.randint(gt_points.shape[0],(3000*multiplier,))
                        surface_points=torch.index_select( gt_points, dim=0, index=rand_indices) 
                        surface_normals=torch.index_select( gt_normals, dim=0, index=rand_indices) 
                        offsurface_points=aabb.rand_points_inside(nr_points=30000*multiplier)
                        # show_points(offsurface_points,"offsurface_points")
                        # show_points(surface_points,"surface_points")

                        points=torch.cat([surface_points, offsurface_points], 0)
                        sdf, sdf_gradients, geom_feat  = model.get_sdf_and_gradient(points, phase.iter_nr)
                        surface_sdf=sdf[:surface_points.shape[0]]
                        surface_sdf_gradients=sdf_gradients[:surface_points.shape[0]]
                        offsurface_sdf=sdf[-offsurface_points.shape[0]:]
                        offsurface_sdf_gradients=sdf_gradients[-offsurface_points.shape[0]:]

                      

                        print("\n phase.iter_nr",  phase.iter_nr)
                       
                        sdf_loss_val=sdf_loss(surface_sdf, surface_sdf_gradients, offsurface_sdf, offsurface_sdf_gradients, surface_normals)
                        loss=sdf_loss_val/30000 #reduce the loss so that the gradient doesn't become too large in the backward pass and it can still be represented with floating value

                

                        print("loss", loss)

                        # if torch.isnan(loss).any():
                            # print("detected nan in loss at iter", phase.iter_nr)
                            # exit()



                        #if its the first time we do a forward on the model we need to create here the optimizer because only now are all the tensors in the model instantiated
                        if first_time:
                            first_time=False
                            params=[]
                            # model_params_without_dense_grid=[]
                            model_params_without_mip_trainable=[]
                            for name, param in model.named_parameters():
                                # print(name)
                                if "mip_trainable" in name:
                                    pass
                                else:
                                    model_params_without_mip_trainable.append(param)
                            # exit()
                            params.append( {'params': model.parameters(), 'weight_decay': train_params.weight_decay(), 'lr': train_params.lr()} )
                            # params.append( {'params': model.parameters(), 'weight_decay': 0.1, 'lr': train_params.lr()} )

                            # params.append( {'params': model_params_without_mip_trainable, 'weight_decay': train_params.weight_decay(), 'lr': train_params.lr()} )
                            # params.append( {'params': model.mip_trainable, 'weight_decay': 0.0, 'lr': 3e-3 } )


                            # params.append( {'params': model.dense_grid, 'weight_decay': train_params.weight_decay(), 'lr': 1e-3} )
                            # params.append( {'params': model_params_without_lattice, 'weight_decay': train_params.weight_decay(), 'lr': train_params.lr()} )
                            # wd_list=np.linspace(0.0000001, 0.01, num=model.nr_resolutions)
                            # for i in range(model.nr_resolutions):
                            #     params.append( {'params': model.lattice_values_list[i], 'weight_decay': wd_list[i], 'lr': train_params.lr()} )

                            # optimizer = torch.optim.AdamW (params, amsgrad=False,  betas=(0.9, 0.99), eps=1e-15)
                            optimizer = RAdam(params, betas=(0.9, 0.99), eps=1e-15) #also gets the shingles
                            # optimizer = VectorAdam(params, betas=(0.9, 0.99), eps=1e-15, axis=-1) #also gets the shingles
                            # optimizer = Adan(params, eps=1e-15) #also gets the shingles
                            # optimizer = apex.optimizers.FusedAdam(params, adam_w_mode=True) #1.7ms for step instead of 3.4 of the radam optimizer
                            # optimizer = RAdam(params) #also gets the shingles
                            # optimizer = LazyAdam(params)
                            # optimizer = torch.optim.Adamax (params)
                            # scheduler_lr_decay=LinearLR(optimizer, start_factor=1.0, end_factor=0.001, total_iters=10000)

                            # scaler = GradScaler()

                        eikonal_loss = torch.abs(offsurface_sdf_gradients.norm(dim=-1) - 1)
                        eikonal_loss=eikonal_loss.mean()

                        cb.after_forward_pass(loss=loss.item(), loss_rgb=0, loss_sdf_surface_area=0, loss_eikonal=eikonal_loss, loss_curvature=0, loss_sdf_grad=0, neus_variance_mean=0, phase=phase, lr=optimizer.param_groups[0]["lr"]) #visualizes the prediction 
                        print("lr ", optimizer.param_groups[0]["lr"])

                    #update gui
                    # ngp_gui.m_c2f_progress=model.c2f.get_last_t()


                    #backward
                    if is_training:
                        optimizer.zero_grad()
                        cb.before_backward_pass()
                        # TIME_START("backward")
                        loss.backward()
                        # scaler.scale(loss).backward()
                        # TIME_END("backward")

                        # summary(model)
                        # if phase.iter_nr==2:
                            # summary(model)
                            # exit(1)

                        grad_clip=20
                        torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=grad_clip, norm_type=2.0)

                        # exit(1)

                        cb.after_backward_pass()
                        optimizer.step()
                        # scaler.step(optimizer)
                        # scaler.update()


                        # scheduler_lr_decay.step()

                        # eixt(1)

                    # TIME_END("FW_BW")

                with torch.set_grad_enabled(False):
                    model.eval()

                    # #visualize every frame
                    # if phase.iter_nr%1==0:
                    #     Gui.show(frame.rgb_32f, "rgb")
                    #     frustum_mesh=frame.create_frustum_mesh(0.02)
                    #     Scene.show(frustum_mesh, "frustum")
                    #     if (train_params.dataset_name()=="volref"):
                    #         cloud=frame_depth.depth2world_xyz_mesh()
                    #         frame.assign_color(cloud) #project the cloud into this frame and creates a color matrix for it
                    #         Scene.show(cloud, "cloud")

                    #save checkpoint
                    if (phase.iter_nr%5000==0 or phase.iter_nr==1) and train_params.save_checkpoint():
                    # if (phase.iter_nr%1000==0 or phase.iter_nr==1) and save_checkpoint:
                        # root_folder=os.path.join( os.path.dirname(os.path.abspath(__file__))  , "../") #points at the root of hair_recon package
                        root_folder=train_params.checkpoint_path()
                        print("saving checkpoint at ", train_params.checkpoint_path())
                        print("experiment name is",experiment_name)
                        model.save(root_folder, experiment_name, phase.iter_nr)


                    ###visualize every once in a while
                    if phase.iter_nr%100==0 or phase.iter_nr==1 or ngp_gui.m_control_view:


                        vis_width=200
                        vis_height=200
                        if first_time_getting_control or ngp_gui.m_control_view:
                            first_time_getting_control=False
                            frame=Frame()
                            frame.from_camera(view.m_camera, vis_width, vis_height)
                            frustum_mesh=frame.create_frustum_mesh(0.1)
                            Scene.show(frustum_mesh,"frustum_mesh_vis")

                        #forward all the pixels
                        ray_origins, ray_dirs=create_rays_from_frame(frame, rand_indices=None) # ray origins and dirs as nr_pixels x 3
                        ray_points_entry, ray_t_entry, ray_points_exit, ray_t_exit, is_hit_valid=aabb.ray_intersection(ray_origins, ray_dirs)
                        # show_points(ray_points_entry, "ray_points_entry", color=[0.0, 1.0, 0.0])
                        # show_points(ray_points_exit, "ray_points_exit", color=[1.0, 0.0, 0.0])



                        # TIME_START("sphere trace")
                        ray_end, ray_end_sdf, ray_end_gradient, ray_end_t=sphere_trace(20, ray_points_entry, ray_origins, ray_dirs, model, phase.iter_nr, 0.9, True, sdf_clamp=0.05)
                        ray_end_converged, ray_end_gradient_converged=filter_unconverged_points(ray_end, ray_end_sdf, ray_end_gradient) #leaves only the points that are converged
                        ray_end_normal=F.normalize(ray_end_gradient, dim=1)
                        ray_end_normal_vis=(ray_end_normal+1.0)*0.5
                        show_points(ray_end, "ray_end", color_per_vert=ray_end_normal_vis, normal_per_vert=ray_end_normal)
                        ray_end_normal=F.normalize(ray_end_gradient_converged, dim=1)
                        ray_end_normal_vis=(ray_end_normal+1.0)*0.5
                        ray_end_normal_tex=ray_end_normal_vis.view(vis_width, vis_height, 3)
                        ray_end_normal_img=tex2img(ray_end_normal_tex)
                        Gui.show(tensor2mat(ray_end_normal_img), "ray_end_normal_img")
                        # TIME_END("sphere trace")

                        #get lso gradients analytically to compare
                        # TIME_START("grad_autograd")
                        # sdf, ray_end_gradient, feat, sdf_residual=model.get_sdf_and_gradient(ray_end.detach(), phase.iter_nr,method="autograd")
                        # TIME_END("grad_autograd")
                        # TIME_START("grad_finite")
                        sdf, ray_end_gradient, feat=model.get_sdf_and_gradient(ray_end.detach(), phase.iter_nr )
                        # TIME_END("grad_finite")
                        ray_end_converged, ray_end_gradient_converged=filter_unconverged_points(ray_end, ray_end_sdf, ray_end_gradient)
                        ray_end_normal=F.normalize(ray_end_gradient, dim=1)
                        ray_end_normal_vis=(ray_end_normal+1.0)*0.5
                        ray_end_normal=F.normalize(ray_end_gradient_converged, dim=1)
                        ray_end_normal_vis=(ray_end_normal+1.0)*0.5
                        ray_end_normal_tex=ray_end_normal_vis.view(vis_width, vis_height, 3)
                        ray_end_normal_img=tex2img(ray_end_normal_tex)
                        Gui.show(tensor2mat(ray_end_normal_img), "ray_end_normal_autograd_img")


                        #show a certain layer
                        layer_width=300
                        layer_height=300
                        x_coord= torch.arange(layer_width).view(-1, 1, 1).repeat(1,layer_height, 1) #width x height x 1
                        z_coord= torch.arange(layer_height).view(1, -1, 1).repeat(layer_width, 1, 1) #width x height x 1
                        zeros=torch.zeros(layer_width, layer_height).view(layer_width, layer_height, 1)
                        x_coord=x_coord/layer_width-0.5
                        z_coord=z_coord/layer_height-0.5
                        # x_coord=x_coord*2.0
                        # z_coord=z_coord*2.0
                        point_layer=torch.cat([x_coord, zeros, z_coord],2).transpose(0,1).reshape(-1,3).cuda()
                        sdf, sdf_gradients, feat = model.get_sdf_and_gradient(point_layer, phase.iter_nr)
                        # sdf_color=sdf.abs().repeat(1,3)
                        # print("sdf_gradients",sdf_gradients)
                        # sdf_color=(sdf_gradients+1.0)*0.5
                        # sdf_color=colormap(sdf.abs())
                        sdf_color=colormap(sdf+0.5, "seismic") #we add 0.5 so as to put the zero of the sdf at the center of the colormap
                        # print("sdf_color", sdf_color.shape)
                        show_points(point_layer, "point_layer", color_per_vert=sdf_color)








                # if phase.loader.is_finished():
                #     cb.epoch_ended(phase=phase, model=model, save_checkpoint=train_params.save_checkpoint(), checkpoint_path=train_params.checkpoint_path() ) 
                #     cb.phase_ended(phase=phase) 
                    # phase.loader.reset()


                if train_params.with_viewer():
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