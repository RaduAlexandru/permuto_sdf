#!/usr/bin/env python3

#This script renders images from our trained model from an arbitrayr frame, so either a controlable fframe or a frame from the dataloader. 
#is just puts the images to screen so you can just click on save if you like it
#this is supposed to be used in a more interactive way while create_my_images is supposed to run remotelly



###CALL with 
#modify the checkpoint_path
# ./permuto_sdf_py/experiments/visualization/render_from_frame.py --dataset dtu --scene dtu_scan63 --comp_name comp_1  [--with_mask]



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
from permuto_sdf_py.utils.permuto_sdf_utils import load_from_checkpoint
from permuto_sdf_py.train_permuto_sdf import run_net_in_chunks
from permuto_sdf_py.train_permuto_sdf import HyperParamsPermutoSDF
import permuto_sdf_py.paths.list_of_checkpoints as list_chkpts
import permuto_sdf_py.paths.list_of_training_scenes as list_scenes





config_file="train_permuto_sdf.cfg"

torch.manual_seed(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
torch.set_grad_enabled(False)
config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../../../config', config_file)



def clamp(num, min_value, max_value):
    num = max(min(num, max_value), min_value)
    return num



def create_sdf_isolines(view, ngp_gui, frame, model, aabb):

    cam=Camera()
    cam.from_frame(frame)

    #get the camera position, create a plane consisting of points at that depth and evaluate the sdf on it
    #make a plane with no rotation
    layer_size=300
    layer_z_coord=ngp_gui.m_isolines_layer_z_coord
    if ngp_gui.m_compute_full_layer or ngp_gui.m_render_full_img:
        layer_size=3000


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
    parser.add_argument('--dataset', required=True,  default="",  help="dataset which can be dtu or bmvs")
    parser.add_argument('--scene', required=True, help='Scene name like dtu_scan24')
    parser.add_argument('--comp_name', required=True,  help='Tells which computer are we using which influences the paths for finding the data')
    parser.add_argument('--with_mask', action='store_true', help="Set this to true in order to train with a mask")
    args = parser.parse_args()
    hyperparams=HyperParamsPermutoSDF()


    #get the results path which will be at the root of the permuto_sdf package 
    permuto_sdf_root=os.path.dirname(os.path.abspath(permuto_sdf.__file__))
    results_path=os.path.join(permuto_sdf_root, "results")
    os.makedirs(results_path, exist_ok=True)
    # ckpts
    checkpoint_path=os.path.join(permuto_sdf_root, "checkpoints/serial_train")


    #####PARAMETERS#######
    with_viewer=True
    print("args.dataset", args.dataset)
    print("args.with_mask", args.with_mask)
    print("results_path",results_path)
    print("with_viewer", with_viewer)
    chunk_size=1000
    iter_nr_for_anneal=9999999
    cos_anneal_ratio=1.0
    low_res=False
    first_time_getting_control=True

    aabb = create_bb_for_dataset(args.dataset)


    #params for rendering
    model_sdf=SDF(in_channels=3, boundary_primitive=aabb, geom_feat_size_out=hyperparams.sdf_geom_feat_size, nr_iters_for_c2f=hyperparams.sdf_nr_iters_for_c2f).to("cuda")
    model_rgb=RGB(in_channels=3, boundary_primitive=aabb, geom_feat_size_in=hyperparams.sdf_geom_feat_size, nr_iters_for_c2f=hyperparams.rgb_nr_iters_for_c2f).to("cuda")
    model_bg=NerfHash(4, boundary_primitive=aabb, nr_iters_for_c2f=hyperparams.background_nr_iters_for_c2f ).to("cuda") 
    if hyperparams.use_occupancy_grid:
        occupancy_grid=OccupancyGrid(256, 1.0, [0,0,0])
    else:
        occupancy_grid=None
    model_sdf.train(False)
    model_rgb.train(False)
    model_bg.train(False)

    
    #get the list of checkpoints
    config_training="with_mask_"+str(args.with_mask) 
    scene_config=args.dataset+"_"+config_training
    ckpts=list_chkpts.ckpts[scene_config]
    ckpt_for_scene=ckpts[args.scene]
    ckpt_path_full=os.path.join(checkpoint_path,ckpt_for_scene,"models")
    #load
    load_from_checkpoint(ckpt_path_full, model_sdf, model_rgb, model_bg, occupancy_grid)



    if with_viewer:
        view=Viewer.create(config_path)
        ngp_gui=NGPGui.create(view)
        view.m_camera.from_string("-0.585619  0.146561  0.708744 -0.0464464  -0.360972 -0.0180037 0.931245 0.0388294 0.0539103 0.0242866 60 0.0502494 5024.94")

    
    loader_train, loader_test= create_dataloader(config_path, args.dataset, args.scene, low_res, args.comp_name, args.with_mask)

    # mesh=Mesh("/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/outputs/output_ingp_meshes/easypbr/head.ply") #ingp
    # mesh.model_matrix.rotate_axis_angle([0,1,0],90) #so that it points accoring to the 3 point light setup
    #transofmr to easypbr
    # if dataset=="dtu":
    #     tf_easypbr_dtu=loader_test.get_tf_easypbr_dtu()
    #     mesh.transform_model_matrix(tf_easypbr_dtu.to_double())
    #     mesh.apply_model_matrix_to_cpu(True)



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
        #get a frame from the dataset
        frame_dataset=loader_train.get_frame_at_idx( clamp(ngp_gui.m_frame_idx_from_dataset, 0, loader_train.nr_samples()-1)  )
        print("frame idx is ", frame_dataset.frame_idx)
        print("cam id is ", frame_dataset.cam_id)
        if(ngp_gui.m_use_controlable_frame):
            frame_to_render=frame_controlable
        else:
            frame_to_render=frame_dataset
            #downsample if we are not rendering on the full img
            if(not ngp_gui.m_render_full_img):
                while frame_to_render.width>100:
                    frame_to_render=frame_to_render.subsample(2.0, True)
            else:
                frame_to_render=frame_to_render.upsample(2.0, False) #make a abig iamge if are rendering the full frame 
        print('frame_to_render',frame_to_render.height," ", frame_to_render.width)
        

        ######RENDER##############################
        chunk_size=ngp_gui.m_chunk_size

        #from model-----------
        pred_rgb_img, pred_rgb_bg_img, pred_normals_img, pred_weights_sum_img=run_net_in_chunks(frame_to_render, chunk_size, args, hyperparams, model_sdf, model_rgb, model_bg, occupancy_grid, iter_nr_for_anneal, cos_anneal_ratio, hyperparams.forced_variance_finish) 
        #view rgb
        pred_rgba_img=torch.cat([pred_rgb_img,pred_weights_sum_img],1)
        pred_img_mat=tensor2mat(pred_rgba_img.detach()).rgba2bgra().to_cv8u()
        Gui.show(pred_img_mat,"pred_img_mat"+str(ngp_gui.m_render_full_img))
        #normals
        pred_normals_img=torch.nn.functional.normalize(pred_normals_img, dim=1)
        pred_normals_img_vis=(pred_normals_img+1.0)*0.5
        pred_normals_img_vis=torch.cat([pred_normals_img_vis,pred_weights_sum_img],1) #concat alpha
        pred_normals_mat=tensor2mat(pred_normals_img_vis.detach()).rgba2bgra().to_cv8u()
        Gui.show(pred_normals_mat,"pred_normals_mat"+str(ngp_gui.m_render_full_img))

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