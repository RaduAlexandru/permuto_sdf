#!/usr/bin/env python3

import torch
import torch.nn.functional as F

import sys
import os
import numpy as np
import time as time_module
import natsort 

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
from permuto_sdf_py.utils.common_utils import show_points
from permuto_sdf_py.utils.common_utils import tex2img
from permuto_sdf_py.utils.common_utils import colormap
from permuto_sdf_py.utils.common_utils import map_range_val
from permuto_sdf_py.utils.aabb import AABB

from permuto_sdf_py.callbacks.callback import *
from permuto_sdf_py.callbacks.viewer_callback import *
from permuto_sdf_py.callbacks.visdom_callback import *
from permuto_sdf_py.callbacks.tensorboard_callback import *
from permuto_sdf_py.callbacks.state_callback import *
from permuto_sdf_py.callbacks.phase import *


config_file="train_sdf_from_mesh.cfg"

torch.manual_seed(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../../../config', config_file)


        


def run():
    # #initialize the parameters used for training
    train_params=TrainParams.create(config_path)
    with_viewer=True    
    if with_viewer:
        view=Viewer.create(config_path)
        ngp_gui=NGPGui.create(view)
        # view.m_camera.from_string("-0.837286  0.360068  0.310824 -0.0496414    -0.5285  -0.030986 0.846901   0.11083  0.235897 -0.152857 60 0.0502494 5024.94")
        view.m_camera.from_string("  1.22547  0.141972 0.0444142 -0.0434999   0.704624  0.0433619 0.706879  0.00373496 -0.00896254   0.0403725 53.929 0.0502494 5024.94")



    #create bounding box for the scene 
    # aabb=Sphere(0.5, [0,0,0])
    aabb=AABB(bounding_box_sizes_xyz=[1.0, 1.0, 1.0], bounding_box_translation=[0.0, 0.0, 0.0])
    


    #model 
    model=SDF(in_channels=4, boundary_primitive=aabb, geom_feat_size_out=0, nr_iters_for_c2f=3000).to("cuda")
    model.train(True)
    #loadchkpt
    cur_dir=os.path.dirname(os.path.abspath(__file__))
    package_root=os.path.join(cur_dir,"../../../")
    # chkpt_path=os.path.join(package_root, "checkpoints/4d/400000/models/sdf_model.pt")
    # chkpt_path=os.path.join(package_root, "checkpoints/4d_v2/300000/models/sdf_model.pt")
    chkpt_path=os.path.join(package_root, "checkpoints/4d_v3/265000/models/sdf_model.pt")
    model.load_state_dict(torch.load(chkpt_path))


    first_time_getting_control=True


    #render continously, and the first X frames, we move the time forward and render to file
    nr_frames=100
    frame_nr=0
    while True:
        with torch.set_grad_enabled(False):
        
            #get time_val
            time_val=map_range_val(frame_nr, 0.0, nr_frames, 0.0, 1.0)


            vis_width=800
            vis_height=510
            if first_time_getting_control or ngp_gui.m_control_view:
                first_time_getting_control=False
                frame=Frame()
                frame.from_camera(view.m_camera, vis_width, vis_height)
                frustum_mesh=frame.create_frustum_mesh(0.1)
                Scene.show(frustum_mesh,"frustum_mesh_vis")

            #forward all the pixels
            ray_origins, ray_dirs=create_rays_from_frame(frame, rand_indices=None) # ray origins and dirs as nr_pixels x 3
            

            #sphere trace those pixels
            ray_end, ray_end_sdf, ray_end_gradient, geom_feat_end, traced_samples_packed=sphere_trace(40, ray_origins, ray_dirs, model, return_gradients=True, sdf_multiplier=0.3, sdf_converged_tresh=0.002, time_val=time_val)
            ray_end_converged, ray_end_gradient_converged, is_converged=filter_unconverged_points(ray_end, ray_end_sdf, ray_end_gradient) #leaves only the points that are converged
            ray_end_normal=F.normalize(ray_end_gradient, dim=1)
            ray_end_normal_vis=(ray_end_normal+1.0)*0.5
            show_points(ray_end, "ray_end", color_per_vert=ray_end_normal_vis, normal_per_vert=ray_end_normal)
            ray_end_normal=F.normalize(ray_end_gradient_converged, dim=1)
            ray_end_normal_vis=(ray_end_normal+1.0)*0.5
            ray_end_normal_tex=ray_end_normal_vis.view(vis_height, vis_width, 3)
            ray_end_normal_img=tex2img(ray_end_normal_tex)
            Gui.show(tensor2mat(ray_end_normal_img), "ray_end_normal_img")
            #create an alpha mask for the normals
            is_converged_tex=is_converged.view(vis_height, vis_width, 1)
            is_converged_img=tex2img(is_converged_tex)*1.0
            ray_end_normal_img_alpha=torch.cat([ray_end_normal_img,is_converged_img],1)
            Gui.show(tensor2mat(ray_end_normal_img_alpha), "ray_end_normal_img_alpha")

            #write to file 
            if(frame_nr<nr_frames):
                out_img_path=os.path.join(package_root,"results/4d_v3/")
                if not os.path.exists(out_img_path):
                    print("path does not exists: ", out_img_path )
                    exit(1)
                tensor2mat(ray_end_normal_img_alpha).to_cv8u().to_file(os.path.join(out_img_path,str(frame_nr)+".png"))
                print("rendering to ", out_img_path, " frame_nr ", frame_nr)


            frame_nr+=1              

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