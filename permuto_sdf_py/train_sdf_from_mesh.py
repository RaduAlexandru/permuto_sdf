#!/usr/bin/env python3

#train a sdf given a known mesh
#CALL with ./permuto_sdf_py/train_sdf_from_mesh.py --exp_info <info>

import torch
import torch.nn.functional as F

import sys
import os
import numpy as np
import argparse

import easypbr
from easypbr  import *

from permuto_sdf  import TrainParams
from permuto_sdf  import NGPGui
from permuto_sdf  import OccupancyGrid
from permuto_sdf  import Sphere
from permuto_sdf_py.models.models import SDF
from permuto_sdf_py.utils.sdf_utils import sdf_loss
from permuto_sdf_py.utils.sdf_utils import sphere_trace
from permuto_sdf_py.utils.sdf_utils import filter_unconverged_points
from permuto_sdf_py.utils.nerf_utils import create_rays_from_frame
from permuto_sdf_py.utils.common_utils import lin2nchw
from permuto_sdf_py.utils.common_utils import show_points
from permuto_sdf_py.utils.common_utils import tex2img
from permuto_sdf_py.utils.common_utils import colormap
from permuto_sdf_py.utils.common_utils import rotate_normals_to_cam_frame
from permuto_sdf_py.utils.aabb import AABB

from permuto_sdf_py.callbacks.callback_utils import *



config_file="train_sdf_from_mesh.cfg"

torch.manual_seed(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../config', config_file)


with_viewer=True
lr=1e-3
# lr=3e-4


def run():
    #argparse
    parser = argparse.ArgumentParser(description='Train sdf')
    parser.add_argument('--exp_info', default="", help='Experiment info string useful for distinguishing one experiment for another')
    args = parser.parse_args()


    # #initialize the parameters used for training
    train_params=TrainParams.create(config_path)    
    if with_viewer:
        view=Viewer.create(config_path)
        ngp_gui=NGPGui.create(view)
        view.m_camera.from_string("-0.837286  0.360068  0.310824 -0.0496414    -0.5285  -0.030986 0.846901   0.11083  0.235897 -0.152857 60 0.0502494 5024.94")
        #  -1.4955 0.889779  2.06755  -0.144471  -0.219777 -0.0329372 0.964226 -0.359422 0.0863527 -0.294206 48.214 0.0502494 5024.94

    experiment_name="sdf_from_mesh"
    if args.exp_info:
        experiment_name+="_"+args.exp_info


    #create bounding box for the scene 
    # aabb=Sphere(0.5, [0,0,0])
    aabb=AABB(bounding_box_sizes_xyz=[1.0, 1.0, 1.0], bounding_box_translation=[0.0, 0.0, 0.0])

    cur_dir=os.path.dirname(os.path.abspath(__file__))
    package_root=os.path.join(cur_dir,"../")
    checkpoint_path=os.path.join(package_root, "checkpoints")
    
   

    #get the mesh for which we will compute the sdf
    mesh=Mesh(  os.path.join(os.path.dirname(easypbr.__file__),  "data/scan_the_world/masterpiece-goliath-ii.stl" ) )
    mesh.model_matrix.rotate_axis_angle([1,0,0],-90)
    mesh.model_matrix.rotate_axis_angle([0,1,0],-120)
    mesh.apply_model_matrix_to_cpu(True)
    mesh.normalize_size()
    mesh.normalize_position()
    mesh.scale_mesh(0.6) #a bit smaller than the bounding box
    mesh.remove_unreferenced_verts()
    mesh.recalculate_normals()
    # mesh.upsample(2,True)
    Scene.show(mesh, "mesh")
    #prepare point and normal
    gt_points=torch.from_numpy(mesh.V.copy()).cuda().float()
    gt_normals=torch.from_numpy(mesh.NV.copy()).cuda().float()



    
    cb=create_callbacks(with_viewer, train_params, experiment_name, config_path)
    phases= [ Phase('train', None, grad=True) ] 
    phase=phases[0] #we usually switch between training and eval phases but here we only train 


    #model 
    model=SDF(in_channels=3, boundary_primitive=aabb, geom_feat_size_out=0, nr_iters_for_c2f=5000).to("cuda")
    model.train(True)

    #optimizer
    optimizer = torch.optim.AdamW (model.parameters(), amsgrad=False,  betas=(0.9, 0.99), eps=1e-15, weight_decay=0.0, lr=lr)

    first_time_getting_control=True

   
    while True:
        cb.before_forward_pass() #sets the appropriate sigma for the lattice


        #sample some random point on the surface and off the surface
        multiplier=1
        rand_indices=torch.randint(gt_points.shape[0],(3000*multiplier,))
        surface_points=torch.index_select( gt_points, dim=0, index=rand_indices) 
        surface_normals=torch.index_select( gt_normals, dim=0, index=rand_indices) 
        offsurface_points=aabb.rand_points_inside(nr_points=30000*multiplier)
        points=torch.cat([surface_points, offsurface_points], 0)

        #run the points through the model
        sdf, sdf_gradients, geom_feat  = model.get_sdf_and_gradient(points, phase.iter_nr)
        surface_sdf=sdf[:surface_points.shape[0]]
        surface_sdf_gradients=sdf_gradients[:surface_points.shape[0]]
        offsurface_sdf=sdf[-offsurface_points.shape[0]:]
        offsurface_sdf_gradients=sdf_gradients[-offsurface_points.shape[0]:]

        

        print("phase.iter_nr",  phase.iter_nr)
        sdf_loss_val=sdf_loss(surface_sdf, surface_sdf_gradients, offsurface_sdf, offsurface_sdf_gradients, surface_normals)
        loss=sdf_loss_val/30000 #reduce the loss so that the gradient doesn't become too large in the backward pass and it can still be represented with floating value
        print("loss", loss)


        #debug
        # print("sdf is ", sdf.min(), sdf.max()) 
        # print("sdf_gradients is ", sdf_gradients.min(), sdf_gradients.max()) 
        # print("surface_normals", surface_normals.min(), surface_normals.max())

        # if torch.isnan(loss):
        #     print("loss is ", loss)
        #     exit(1)

      

        cb.after_forward_pass(phase=phase, loss=loss.item(), lr=optimizer.param_groups[0]["lr"]) #visualizes the prediction 

        #update gui
        if with_viewer:
            ngp_gui.m_c2f_progress=model.c2f.get_last_t()


        #backward
        optimizer.zero_grad()
        cb.before_backward_pass()
        loss.backward()


        cb.after_backward_pass()
        optimizer.step()


        if train_params.save_checkpoint() and (phase.iter_nr%5000==0 or phase.iter_nr==1):
            model.save(checkpoint_path, experiment_name, phase.iter_nr)


        #visualize the sdf by sphere tracing
        if phase.iter_nr%100==0 or phase.iter_nr==1 or ngp_gui.m_control_view:
            with torch.set_grad_enabled(False):
                model.eval()


                vis_width=500
                vis_height=500
                if first_time_getting_control or ngp_gui.m_control_view:
                    first_time_getting_control=False
                    frame=Frame()
                    frame.from_camera(view.m_camera, vis_width, vis_height)
                    frustum_mesh=frame.create_frustum_mesh(0.1)
                    Scene.show(frustum_mesh,"frustum_mesh_vis")

                #forward all the pixels
                ray_origins, ray_dirs=create_rays_from_frame(frame, rand_indices=None) # ray origins and dirs as nr_pixels x 3
                


                #sphere trace those pixels
                ray_end, ray_end_sdf, ray_end_gradient, geom_feat_end, traced_samples_packed=sphere_trace(20, ray_origins, ray_dirs, model, return_gradients=True, sdf_multiplier=0.9, sdf_converged_tresh=0.0002)
                ray_end_converged, ray_end_gradient_converged, is_converged=filter_unconverged_points(ray_end, ray_end_sdf, ray_end_gradient) #leaves only the points that are converged
                ray_end_normal=F.normalize(ray_end_gradient, dim=1)
                ray_end_normal_img=lin2nchw(ray_end_normal, frame.height, frame.width)
                ray_end_normal_vis=(ray_end_normal+1.0)*0.5
                show_points(ray_end, "ray_end", color_per_vert=ray_end_normal_vis, normal_per_vert=ray_end_normal)
                ray_end_normal=F.normalize(ray_end_gradient_converged, dim=1)
                ray_end_normal_vis=(ray_end_normal+1.0)*0.5
                ray_end_normal_tex=ray_end_normal_vis.view(vis_height, vis_width, 3)
                ray_end_normal_img_vis=tex2img(ray_end_normal_tex)
                Gui.show(tensor2mat(ray_end_normal_img_vis).rgb2bgr(), "ray_end_normal_img_vis")
                #normals view_coords
                ray_end_normal_viewcoords=rotate_normals_to_cam_frame(ray_end_normal_img, frame)
                is_converged_img=lin2nchw(is_converged, frame.height, frame.width)
                ray_end_normal_viewcoords[~is_converged_img.bool().repeat(1,3,1,1)]=0.0
                ray_end_normal_viewcoords_vis=(ray_end_normal_viewcoords+1.0)*0.5
                Gui.show(tensor2mat(ray_end_normal_viewcoords_vis).rgb2bgr(), "ray_end_normal_img_viewcoords")

        if phase.iter_nr%100==0 or phase.iter_nr==1:
            with torch.set_grad_enabled(False):
                #show a certain layer of the SDF
                layer_width=300
                layer_height=300
                x_coord= torch.arange(layer_width).view(-1, 1, 1).repeat(1,layer_height, 1) #width x height x 1
                z_coord= torch.arange(layer_height).view(1, -1, 1).repeat(layer_width, 1, 1) #width x height x 1
                zeros=torch.zeros(layer_width, layer_height).view(layer_width, layer_height, 1)
                x_coord=x_coord/layer_width-0.5
                z_coord=z_coord/layer_height-0.5
                point_layer=torch.cat([x_coord, zeros, z_coord],2).transpose(0,1).reshape(-1,3).cuda()
                sdf, sdf_gradients, feat = model.get_sdf_and_gradient(point_layer, phase.iter_nr)
                sdf_color=colormap(sdf+0.5, "seismic") #we add 0.5 so as to put the zero of the sdf at the center of the colormap
                show_points(point_layer, "point_layer", color_per_vert=sdf_color)



        #finally just update the opengl viewer
        with torch.set_grad_enabled(False):
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