#!/usr/bin/env python3


#This script creates overview image for the paper


###CALL with 
# ./permuto_sdf_py/experiments/figures/create_overview_image.py 


import torch

import sys
import os
import numpy as np
from tqdm import tqdm
import time
import torchvision
import argparse
import math

import easypbr
from easypbr  import *
from dataloaders import *

import permuto_sdf
from permuto_sdf  import NGPGui
from permuto_sdf  import TrainParams
from permuto_sdf  import OccupancyGrid
from permuto_sdf_py.utils.aabb import AABB
from permuto_sdf_py.models.models import SDF
from permuto_sdf_py.models.models import RGB
from permuto_sdf_py.models.models import NerfHash
from permuto_sdf_py.models.models import Colorcal
from permuto_sdf_py.utils.common_utils import map_range_tensor
from permuto_sdf_py.utils.common_utils import map_range_np
from permuto_sdf_py.utils.common_utils import create_dataloader
from permuto_sdf_py.utils.common_utils import create_bb_for_dataset
from permuto_sdf_py.utils.common_utils import create_bb_mesh
from permuto_sdf_py.utils.common_utils import nchw2lin
from permuto_sdf_py.utils.common_utils import lin2nchw
from permuto_sdf_py.utils.sdf_utils import extract_mesh_from_sdf_model
from permuto_sdf_py.utils.permuto_sdf_utils import load_from_checkpoint
from permuto_sdf_py.train_permuto_sdf import run_net_in_chunks
from permuto_sdf_py.train_permuto_sdf import HyperParamsPermutoSDF
import permuto_sdf_py.paths.list_of_checkpoints as list_chkpts
import permuto_sdf_py.paths.list_of_training_scenes as list_scenes




torch.manual_seed(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)

def make_camera_frustum(idx, res_multiplier):

    if idx==0: #captured at   3.4534 0.776799  0.23304 -0.0880444   0.670344  0.0805919 0.732372   0.630147  0.0852987 -0.0170724 30 0.0648786 6487.86
        #position
        pos=[1.5, 0.3, 0.1]
        lookat=[0,-0.07,0]
        dir_shift=[0, -0.04, -0.05]
        ray_dist=1.7
        #get color from a mat
        img_mat=Mat("/media/rosu/Data/phd/c_ws/src/permuto_sdf/recordings/overview/img1_crop_shift.png")
        # img_mat_mask=Mat("/media/rosu/Data/data/neus_data/data_DTU/dtu_scan122/mask/062.png")
        img_t=mat2tensor(img_mat, True)
        print("img_t",img_t.type())
        # img_t=torch.clamp(img_t*1.5, 0, 255).byte() #incrase brightness
        # img_t_mask=mat2tensor(img_mat_mask, True)
        # img_t=torch.cat([img_t,img_t_mask[:,0:1,:,:]],1)
        img_mat=tensor2mat(img_t).rgba2bgra()
        img_mat.set_alpha_to_value(255)
    elif idx==1: #captured at  0.667222 -0.401567  -3.38892 0.00464816   0.994293 -0.0488266 0.0946479   0.12228 -0.117219  -0.55243 30 0.0648786 6487.86
        #position
        pos=[0.3, -0.2, -1.3]
        lookat=[0.075,-0.07,0]
        dir_shift=[0.05, 0, 0]
        ray_dist=1.5
        #get color from a mat
        img_mat=Mat("/media/rosu/Data/phd/c_ws/src/permuto_sdf/recordings/overview/img2_crop.png")
        # img_mat_mask=Mat("/media/rosu/Data/data/neus_data/data_DTU/dtu_scan122/mask/039.png")
        img_t=mat2tensor(img_mat, True)
        # img_t=img_t*2 #incrase brightness
        # img_t=torch.clamp(img_t*2, 0, 255).byte()
        # img_t_mask=mat2tensor(img_mat_mask, True)
        # img_t=torch.cat([img_t,img_t_mask[:,0:1,:,:]],1)
        img_mat=tensor2mat(img_t).rgba2bgra()
        img_mat.set_alpha_to_value(255)

    #make a frustum
    cam=Camera()
    cam.set_position(pos)
    cam.set_lookat(lookat)
    cam.push_away_by_dist(0.2)
    cam.m_fov=55
    frame=Frame()
    frame.from_camera(cam, 100,100)
    #make a color so that we show a texture on the frustum
    # frustum_img_color=torch.tensor([1.0, 0.8, 0.3])
    # frustum_img_color=torch.tensor([0.3, 0.8, 1.0])
    # frustum_img_tensor=frustum_img_color.view(1,3,1,1)
    # frustum_img_mat=tensor2mat(frustum_img_tensor).rgb2bgr()
    
    # print("img_t",img_t.shape)
    # print("img_t_mask",img_t_mask.shape)
    frame.rgb_32f=img_mat
    #create frustum mesh
    frustum_mesh=frame.create_frustum_mesh(0.25)
    frustum_mesh.m_vis.m_line_width=3.0*res_multiplier
    frustum_mesh.m_vis.m_line_color=[0.1, 0.1, 0.1]
    Scene.show(frustum_mesh, "frustum"+str(idx) )


    #make a line from the frustum towards the center
    p0=cam.position() +cam.direction()*0.2
    p1=cam.position() +(cam.direction()+dir_shift)*ray_dist
    line_mesh=Mesh()
    line_mesh.V=[        #fill up the vertices of the mesh as a matrix of Nx3
        p0,
        p1
    ]
    line_mesh.E=[        #fill up the vertices of the mesh as a matrix of Nx3
        0,1
    ]
    line_mesh.m_vis.m_show_points=False
    line_mesh.m_vis.m_show_lines=True
    # line_mesh.m_vis.m_line_color=[0.1, 0.1, 0.1]
    # line_mesh.m_vis.m_line_color=[0.9, 0.5, 0.15]
    line_mesh.m_vis.m_line_color=[0.15, 0.7, 0.9]
    line_mesh.m_vis.m_line_width=4.0*res_multiplier
    Scene.show(line_mesh,"line_mesh"+str(idx))


def run():

    config_file="create_overview_image.cfg"
    config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../../../config', config_file)

    view=Viewer.create(config_path)
    # view.m_camera.from_string(" 8.91268  3.34954 -5.15932 -0.0780585    0.85163   0.132681 0.501024 -0.0479057  0.0758429  -0.179569 30 0.0648786 6487.86")
    # view.m_camera.from_string("6.85141  2.2075 -4.8223 -0.0588088   0.875751   0.110457 0.466259 -0.0479057  0.0758429  -0.179569 30 0.0648786 6487.86")
    # view.m_camera.from_string("4.43347 1.46044 -3.1952 -0.0588088   0.875751   0.110457 0.466259 -0.0479057  0.0758429  -0.179569 30 0.0648786 6487.86")
    view.m_camera.from_string(" 4.54251  1.38372 -3.06833 -0.0588088   0.875751   0.110457 0.466259    0.0611393 -0.000876501   -0.0527036 30 0.0648786 6487.86")

    res_multiplier=3.0


    #get a mesh to show in a cube,  we need the loader so that we can place it in easypbr coords
    dataset="dtu"
    scan_name="dtu_scan122"
    use_home=True
    use_all_imgs=True
    without_mask=False
    # loader, _= create_dataloader(dataset, scan_name, config_path, use_home, use_all_imgs, without_mask)


    #make cube
    aabb=AABB([1.0, 1.0, 1.0], [0,0,0])
    aabb_mesh=create_bb_mesh(aabb)
    aabb_mesh.m_vis.m_show_wireframe=False
    aabb_mesh.m_vis.m_show_lines=True
    aabb_mesh.m_vis.m_line_color=[0.5, 0.5, 0.5]
    aabb_mesh.m_vis.m_line_width=1.0*res_multiplier
    aabb_mesh.m_vis.m_is_line_dashed=True
    aabb_mesh.m_vis.m_dash_size=20*res_multiplier
    aabb_mesh.m_vis.m_gap_size=20*res_multiplier
    Scene.show(aabb_mesh, "aabb_mesh")

    #mesh
    mesh=Mesh("/media/rosu/Data/data/3d_objs/sketchfab/orchid2/source/model/model_clean.obj")
    # mesh.model_matrix.rotate_axis_angle([0,1,0],-32) #for the orchid2
    mesh.model_matrix.rotate_axis_angle([0,1,0],180-10) #for the orchid2
    mesh.apply_model_matrix_to_cpu(True)
    mesh.normalize_size()
    mesh.normalize_position()
    mesh.model_matrix.translate([0.0, 0.0, -0.1])
    mesh.scale_mesh(0.9) #a bit smaller than the bounding box
    mesh.set_diffuse_tex("/media/rosu/Data/data/3d_objs/sketchfab/orchid2/source/model/tex_u1_v1.jpg")
    Scene.show(mesh, "mesh") 
    #vis
    mesh.m_vis.m_opacity=0.4
    Scene.show(mesh,"mesh")


    make_camera_frustum(0, res_multiplier) 
    make_camera_frustum(1, res_multiplier) 


    #render image
    view.m_viewport_size=[3000,3000]
    view.draw()
    rendered=view.rendered_mat_no_gui(True).rgba2bgra().flip_y()
    Gui.show(rendered,"rendered")


    

    while True:
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

