#!/usr/bin/env python3

#This script just loads the checkpoint for the sdf model and render the isolines

###CALL with 
#modify the checkpoint_path
# ./permuto_sdf_py/experiments/visualization/visualize_sdf_isolines.py --ckpt_path <path_sdf_model>



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
from permuto_sdf_py.utils.common_utils import nchw2lin
from permuto_sdf_py.utils.common_utils import lin2nchw
from permuto_sdf_py.utils.sdf_utils import extract_mesh_from_sdf_model
from permuto_sdf_py.utils.permuto_sdf_utils import load_from_checkpoint
from permuto_sdf_py.train_permuto_sdf import run_net_in_chunks
from permuto_sdf_py.train_permuto_sdf import HyperParamsPermutoSDF
import permuto_sdf_py.paths.list_of_checkpoints as list_chkpts
import permuto_sdf_py.paths.list_of_training_scenes as list_scenes





config_file="visualize_sdf_isolines.cfg"

torch.manual_seed(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
torch.set_grad_enabled(False)
config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../../../config', config_file)




def create_sdf_isolines(view, ngp_gui, frame, model, aabb):

    cam=Camera()
    cam.from_frame(frame)

    #get the camera position, create a plane consisting of points at that depth and evaluate the sdf on it
    #make a plane with no rotation
    layer_size=500
    layer_z_coord=ngp_gui.m_isolines_layer_z_coord
    if ngp_gui.m_compute_full_layer or ngp_gui.m_render_full_img:
        layer_size=4000


    layer_width=layer_size
    layer_height=layer_size
    x_coord= torch.arange(layer_width).view(-1, 1, 1).repeat(1,layer_height, 1) #width x height x 1
    y_coord= torch.arange(layer_height).view(1, -1, 1).repeat(layer_width, 1, 1) #width x height x 1
    z_coord=torch.zeros(layer_width, layer_height).view(layer_width, layer_height, 1)
    z_coord+=layer_z_coord
    x_coord=x_coord/layer_width-0.5
    y_coord=y_coord/layer_height-0.5
    x_coord=x_coord*2.0
    y_coord=y_coord*2.0
    point_layer=torch.cat([x_coord, y_coord, z_coord],2).transpose(0,1).reshape(-1,3).cuda()

    #rotate points based on camera rotation
    points_mesh=Mesh()
    points_mesh.V=point_layer.detach().double().cpu().numpy()
    points_mesh.model_matrix.set_linear( cam.cam_axes() )
    points_mesh.m_vis.m_show_points=True
    points_mesh.apply_model_matrix_to_cpu(False)

    #evaluate the sdf at those points
    chunk_size=6000
    query_pts_flat=torch.from_numpy(points_mesh.V).float().cuda()
    nr_chunks = math.ceil( query_pts_flat.shape[0]/chunk_size)
    query_pts_flat_list=torch.chunk(query_pts_flat, nr_chunks)
    nr_points=query_pts_flat.shape[0]
    query_pts=None #Release the memory
    query_pts_flat=None
    sdf_full = torch.zeros(nr_points)
    for i in range(len(query_pts_flat_list)):
        # print("processing ", i, " of ", len(query_pts_flat_list) )
        pts = query_pts_flat_list[i]
        sdf, feat=model(pts, 9999999) #use a really large iter_nr to ensure that the coarse2fine has finished
        sdf_full[i*pts.shape[0]:(i+1)*pts.shape[0]] = sdf.squeeze(1)
    # sdf_full = sdf_full.cpu().numpy()
    # sdf_full=sdf_full.abs()*10

    #make isolines of for sdf values that are the same
    isoline_width=ngp_gui.m_isoline_width
    distance_between_isolines=ngp_gui.m_distance_between_isolines
    colormngr=ColorMngr()
    # sdf_isolines_colors=sdf_full.view(-1,1).repeat(1,3)
    # sdf_isolines_colors=torch.ones_like(sdf_isolines_colors)
    # sdf_isolines_color_cpu=colormngr.eigen2color((1.0-sdf_full*3).double().cpu().numpy(), "viridis")

    #color with pubu
    x=map_range_tensor(sdf_full, 0.0, 0.3, 0.2, 1.0)
    sdf_isolines_color_cpu=colormngr.eigen2color((x).double().cpu().numpy(), "pubu")

    sdf_isolines_colors=torch.from_numpy(sdf_isolines_color_cpu).cuda().float()
    for i in range(15):
        isoline_center=i*distance_between_isolines
        cur_range_min=isoline_center-isoline_width/2
        cur_range_max=isoline_center+isoline_width/2
        values_in_isorange=torch.logical_and( sdf_full>cur_range_min, sdf_full<cur_range_max )

        #color the lines as black
        sdf_isolines_colors[values_in_isorange,:]=0.0

        #color with pubu
        val_for_line = np.array([isoline_center])
        val_for_line=map_range_np(val_for_line, 0.0, 0.2, 0.3, 1.0)
        color_line= colormngr.eigen2color( val_for_line, "pubu")
        color_line=torch.from_numpy(color_line).view(1,3).float().cuda()
        sdf_isolines_colors[values_in_isorange,:]=color_line


    #set sdf as color
    # points_mesh.C=sdf_full.view(-1,1).repeat(1,3).double().cpu().numpy()
    points_mesh.C=sdf_isolines_colors.double().cpu().numpy()
    points_mesh.m_vis.set_color_pervertcolor()

    #make also some normals because you can then enable ssao for it
    NV= -cam.direction()
    NV=NV.reshape((1,3))
    NV=np.repeat(NV,points_mesh.V.shape[0], axis=0)
    points_mesh.NV=NV


    # points_mesh=aabb.remove_points_outside(points_mesh)
    points=torch.from_numpy(points_mesh.V).float().cuda()
    is_valid=aabb.check_point_inside_primitive(points)
    points_mesh.remove_marked_vertices( is_valid.flatten().bool().cpu().numpy() ,True)
    points_mesh.m_vis.m_show_mesh=False
    if ngp_gui.m_compute_full_layer:
        points_mesh.m_vis.m_point_size=1.0
        Scene.show(points_mesh, "points_mesh_full")
    else:
        Scene.show(points_mesh, "points_mesh")

    # ngp_gui.m_compute_full_layer=False #go back to computing a small layer


   



def render_easypbr_from_frame(view, ngp_gui, frame):
    

    #render img from frame into mat


    #save some previous state
    prev_viewport=view.m_viewport_size.copy()
    prev_cam=view.m_camera

    #set new state
    view.m_viewport_size=[frame.width, frame.height]
    cam=Camera()
    cam.from_frame(frame)
    cam.m_exposure=prev_cam.m_exposure
    view.m_camera=cam

    #draw and get the mat
    view.draw()
    rendered=view.rendered_mat_no_gui(True).rgba2bgra().flip_y()
    # print("downloading meshidgtex")
    # rendered_mesh_id=view.gbuffer_mat_with_name("mesh_id_gtex").clone().flip_y()
    rendered_normals=view.gbuffer_mat_with_name("normal_gtex").flip_y().set_value_to_alpha(0)
    # print("downlaoded meshidgtex")
    # rendered_mesh_id=view.gbuffer_mat_with_name("color_with_transparency_gtex").flip_y()

    #restore
    view.m_viewport_size=prev_viewport
    view.m_camera=prev_cam

    #show
    Gui.show(rendered,"rendered_full"+str(ngp_gui.m_render_full_img))
    Gui.show(rendered_normals,"rendered_normals"+str(ngp_gui.m_render_full_img))


def run():
    #argparse
    parser = argparse.ArgumentParser(description='prepare dtu evaluation')
    parser.add_argument('--ckpt_path', required=True,   help="ckpt_path to the sdf model")
    args = parser.parse_args()
    hyperparams=HyperParamsPermutoSDF()


    


    #####PARAMETERS#######
    with_viewer=True
    chunk_size=1000
    iter_nr_for_anneal=9999999
    cos_anneal_ratio=1.0
    first_time_getting_control=True
    sdf_geom_feat_size=0

    # aabb = create_bb_for_dataset("dtu")
    aabb=AABB(bounding_box_sizes_xyz=[0.6, 0.65, 0.5], bounding_box_translation=[0.0, 0.0, 0.0])
    # -1.43572  1.20796 0.145403 -0.256707 -0.631255 -0.234213 0.693193 -0.0108633 -0.0232055  0.0144082 52.5 0.0502494 5024.94

    #params for rendering
    model_sdf=SDF(in_channels=3, boundary_primitive=aabb, geom_feat_size_out=sdf_geom_feat_size, nr_iters_for_c2f=hyperparams.sdf_nr_iters_for_c2f).to("cuda")
    model_sdf.train(False)


    #load
    model_sdf.load_state_dict(torch.load(args.ckpt_path) )

    

    if with_viewer:
        view=Viewer.create(config_path)
        ngp_gui=NGPGui.create(view)
        #with the rotated rochid at -47 degrees
        view.m_camera.from_string("  -1.39789   1.25715 0.0144385 -0.257307 -0.658095 -0.257305 0.659006 -0.0108633 -0.0232055  0.0144082 52.5 0.0502494 5024.94")
        view.spotlight_with_idx(0).from_string("-1.19069  1.36252 0.164685 -0.310569 -0.598559 -0.270566 0.687067 -3.01493e-05  -0.00322646            0 40 0.181798 181.798")
        ngp_gui.m_isolines_layer_z_coord=-0.057
        ngp_gui.m_distance_between_isolines=0.017



        



    #extract a mesh just to show it
    extracted_mesh=extract_mesh_from_sdf_model(model_sdf, nr_points_per_dim=200, min_val=-0.5, max_val=0.5)
    Scene.show(extracted_mesh, "extracted_mesh")


    #show another mesh if necessary
    # mesh=Mesh("/media/rosu/Data/data/3d_objs/sketchfab/orchid2/source/model/model_clean.obj")
    # # mesh.model_matrix.rotate_axis_angle([0,1,0],-32) #for the orchid2
    # mesh.model_matrix.rotate_axis_angle([0,1,0],-47) #for the orchid2
    # mesh.apply_model_matrix_to_cpu(True)
    # mesh.normalize_size()
    # mesh.normalize_position()
    # mesh.scale_mesh(0.6) #a bit smaller than the bounding box
    # mesh.set_diffuse_tex("/media/rosu/Data/data/3d_objs/sketchfab/orchid2/source/model/tex_u1_v1.jpg")
    # Scene.show(mesh, "mesh")




    while True:
        view.update()

        #get frame controlable
        vis_width=100
        vis_height=100
        if(ngp_gui.m_render_full_img):
            vis_width=2048
            vis_height=2048
            print("render full img", vis_width, " ", vis_height)
        if first_time_getting_control or ngp_gui.m_control_view:
            first_time_getting_control=False
            frame_controlable=Frame()
            print("creating frame controlable", vis_width, " ", vis_height)
            frame_controlable.from_camera(view.m_camera, vis_width, vis_height)
            frustum_mesh_controlable=frame_controlable.create_frustum_mesh(0.1)
            Scene.show(frustum_mesh_controlable,"frustum_mesh_controlable")
        if(ngp_gui.m_use_controlable_frame):
            frame_to_render=frame_controlable
        print('frame_to_render',frame_to_render.height," ", frame_to_render.width)
        
        

        ######RENDER##############################

        #render isolines---------
        create_sdf_isolines(view, ngp_gui, frame_to_render, model_sdf, aabb)

        #render the current frame, from easy into a mat so we can save easily
        render_easypbr_from_frame(view, ngp_gui, frame_to_render)


        #reset all the things for rendering at fuill res
        ngp_gui.m_render_full_img=False
        ngp_gui.m_compute_full_layer=False

        #write cam trajectory to file 
        # cam_traj_file.write(view.m_camera.to_string()+"\n")



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