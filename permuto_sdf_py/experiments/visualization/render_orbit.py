#!/usr/bin/env python3

#This script renders images making an orbit around the scene

###CALL with 
#where ckpt_path is the checkpoints path and the  out_path is where you want to store your images
# ./permuto_sdf_py/experiments/visualization/render_orbit.py --ckpt_path <ckpt_path> [ --out_path <out_path>  --no_viewer] 



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
from permuto_sdf_py.utils.common_utils import rotate_normals_to_cam_frame
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





def run():
    #argparse
    parser = argparse.ArgumentParser(description='prepare dtu evaluation')
    parser.add_argument('--ckpt_path', required=True,  default="",  help="checkpoint_path")
    parser.add_argument('--out_path', default="",  help="out_path")
    parser.add_argument('--no_viewer', action='store_true', help="Set this to true in order disable the viewer")
    parser.add_argument('--with_mask', action='store_true', help="Set this to true in order to train with a mask")
    args = parser.parse_args()
    hyperparams=HyperParamsPermutoSDF()
    with_viewer=not args.no_viewer



    #####PARAMETERS#######
    print("out_path", args.out_path)
    print("with_viewer", with_viewer)
    print("ckpt_path", args.ckpt_path)
    chunk_size=3000
    iter_nr_for_anneal=9999999
    cos_anneal_ratio=1.0

    aabb = create_bb_for_dataset("dtu")


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

    
    #load
    load_from_checkpoint(args.ckpt_path, model_sdf, model_rgb, model_bg, occupancy_grid)



    if with_viewer:
        view=Viewer.create(config_path)
        ngp_gui=NGPGui.create(view)
        view.m_camera.from_string("-0.585619  0.146561  0.708744 -0.0464464  -0.360972 -0.0180037 0.931245 0.0388294 0.0539103 0.0242866 60 0.0502494 5024.94")

    

    cam=Camera()
    cam.set_position([0,0,1])
    cam.set_lookat([0,0,0])
    cam.push_away_by_dist(0.8)
    cam.orbit_x(-30)

    idx=0
    while True:
        print("render frame", idx)

        if with_viewer:
            view.update()
            # view.m_camera=cam

        #get frame controlable
        vis_width=2048
        vis_height=2048
        # vis_width=100
        # vis_height=100
        frame_to_render=Frame()
        frame_to_render.from_camera(cam, vis_width, vis_height)
        cam.orbit_y(1)

        

        # ######RENDER##############################
        pred_rgb_img, pred_rgb_bg_img, pred_normals_img, pred_weights_sum_img=run_net_in_chunks(frame_to_render, chunk_size, args, hyperparams, model_sdf, model_rgb, model_bg, occupancy_grid, iter_nr_for_anneal, cos_anneal_ratio, hyperparams.forced_variance_finish) 
        #view rgb
        pred_rgb_mat=tensor2mat(pred_rgb_img.detach()).rgb2bgr().to_cv8u()
        pred_rgba_img=torch.cat([pred_rgb_img,pred_weights_sum_img],1)
        pred_rgba_mat=tensor2mat(pred_rgba_img.detach()).rgba2bgra().to_cv8u()
        #normals
        pred_normals_img=torch.nn.functional.normalize(pred_normals_img, dim=1)
        pred_normals_img_vis=(pred_normals_img+1.0)*0.5
        pred_normals_img_vis=torch.cat([pred_normals_img_vis,pred_weights_sum_img],1) #concat alpha
        pred_normals_mat=tensor2mat(pred_normals_img_vis.detach()).rgba2bgra().to_cv8u()
        #normal view coords
        pred_normals_img_cam_coords=rotate_normals_to_cam_frame(pred_normals_img, frame_to_render)
        pred_normals_img_cam_coords_vis=(pred_normals_img_cam_coords+1.0)*0.5
        pred_normals_img_cam_coords_vis=torch.cat([pred_normals_img_cam_coords_vis,pred_weights_sum_img],1) #concat alpha
        pred_normals_img_cam_coords_vis_mat=tensor2mat(pred_normals_img_cam_coords_vis.detach()).rgba2bgra().to_cv8u()

        #show
        if with_viewer:
            Gui.show(pred_rgb_mat,"pred_rgb_mat")
            Gui.show(pred_rgba_mat,"pred_rgba_mat")
            Gui.show(pred_normals_mat,"pred_normals_mat")
            Gui.show(pred_normals_img_cam_coords_vis_mat,"pred_normals_img_cam_coords")


        #save to file if needed 
        if args.out_path:
            os.makedirs(os.path.join(args.out_path,"rgb"), exist_ok=True)
            os.makedirs(os.path.join(args.out_path,"rgba"), exist_ok=True)
            os.makedirs(os.path.join(args.out_path,"normal"), exist_ok=True)
            #prepare imgs
            #wrtie to file
            pred_rgb_mat.to_file( os.path.join(args.out_path,"rgb",str(idx)+".png") )
            pred_rgba_mat.to_file( os.path.join(args.out_path,"rgba",str(idx)+".png") )
            pred_normals_img_cam_coords_vis_mat.to_file( os.path.join(args.out_path,"normal",str(idx)+".png") )

        idx+=1
            




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