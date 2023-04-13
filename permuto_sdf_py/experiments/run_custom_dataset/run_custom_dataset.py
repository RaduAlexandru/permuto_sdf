#!/usr/bin/env python3

#this scripts shows how to run PermutoSDF on your own custom dataset
#You would need to modify the function create_custom_dataset() to suit your needs. The current code is setup to read from the easypbr_render dataset (see README.md for the data) but you need to change it for your own data. The main points are that you need to provide an image, intrinsics and extrinsics for each your cameras. Afterwards you need to scale your scene so that your object of interest lies within the bounding sphere of radius 0.5 at the origin.

#CALL with ./permuto_sdf_py/experiments/run_custom_dataset/run_custom_dataset.py --exp_info test [--no_viewer]

import torch
import argparse
import os
import natsort
import numpy as np

import easypbr
from easypbr  import *
from dataloaders import *

import permuto_sdf
from permuto_sdf  import TrainParams
from permuto_sdf_py.utils.common_utils import create_dataloader
from permuto_sdf_py.utils.permuto_sdf_utils import get_frames_cropped
from permuto_sdf_py.train_permuto_sdf import train
from permuto_sdf_py.train_permuto_sdf import HyperParamsPermutoSDF
import permuto_sdf_py.paths.list_of_training_scenes as list_scenes



torch.manual_seed(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)


parser = argparse.ArgumentParser(description='Train sdf and color')
parser.add_argument('--dataset', default="custom", help='Dataset name which can also be custom in which case the user has to provide their own data')
parser.add_argument('--with_mask', action='store_true', help="Set this to true in order to train with a mask")
parser.add_argument('--exp_info', default="", help='Experiment info string useful for distinguishing one experiment for another')
parser.add_argument('--no_viewer', action='store_true', help="Set this to true in order disable the viewer")
args = parser.parse_args()
with_viewer=not args.no_viewer


#MODIFY these for your dataset!
SCENE_SCALE=0.9
SCENE_TRANSLATION=[0,0,0]
IMG_SUBSAMPLE_FACTOR=1.0 #subsample the image to lower resolution in case you are running on a low VRAM GPU. The higher this number, the smaller the images
DATASET_PATH="/media/rosu/Data/data/permuto_sdf_data/easy_pbr_renders/head/" #point this to wherever you downloaded the easypbr_data (see README.md for download link)
 
def create_custom_dataset():
    #CREATE CUSTOM DATASET---------------------------
    #We need to fill easypbr.Frame objects into a list. Each Frame object contains the image for a specific camera together with extrinsics and intrinsics
    #intrinsics and extrinsics
    assert os.path.exists(DATASET_PATH), "The dataset path does not exist. Please point to the path where you downloaded the EasyPBR renders"


    intrinics_extrinsics_file=os.path.join(DATASET_PATH,"poses_and_intrinsics.txt")
    with open(intrinics_extrinsics_file) as file:
        lines = [line.rstrip() for line in file]
    #remove comments
    lines = [item for item in lines if not item.startswith('#')]
    #images
    path_imgs=os.path.join(DATASET_PATH,"imgs_train") #modify this to wherever your 
    imgs_names_list=[img_name for img_name in os.listdir(path_imgs)]
    imgs_names_list=natsort.natsorted(imgs_names_list,reverse=False)

    #create list of frames for this scene
    frames=[]
    for idx, img_name in enumerate(imgs_names_list):
        #load img as single precision RGB
        print("img_name", img_name)
        frame=Frame()
        img=Mat(os.path.join(path_imgs,img_name))
        img=img.to_cv32f()
        if img.channels()==4:
            img=img.rgba2rgb()
        frame.rgb_32f=img

        #img_size
        frame.width=img.cols
        frame.height=img.rows

        #intrinsics as fx, fy, cx, cy
        calib_line=lines[idx]
        calib_line_split=calib_line.split()
        K=np.identity(3)
        K[0][0]=calib_line_split[-4] #fx
        K[1][1]=calib_line_split[-3] #fy
        K[0][2]=calib_line_split[-2] #cx
        K[1][2]=calib_line_split[-1] #cy
        frame.K=K

        #extrinsics as a tf_cam_world (transformation that maps from world to camera coordiantes)
        translation_world_cam=calib_line_split[1:4] #translates from cam to world
        quaternion_world_cam=calib_line_split[4:8] #rotates from cam to world
        tf_world_cam=Affine3f()
        tf_world_cam.set_quat(quaternion_world_cam) #assumes the quaternion is expressed as [qx,qy,qz,qw]
        tf_world_cam.set_translation(translation_world_cam)
        tf_cam_world=tf_world_cam.inverse() #here we get the tf_cam_world that we need
        frame.tf_cam_world=tf_cam_world
        #ALTERNATIVELLY if you have already the extrinsics as a numpy matrix you can use the following line
        # frame.tf_cam_world.from_matrix(YOUR_4x4_TF_CAM_WORLD_NUMPY_MATRIX) 

        #scale scene so that the object of interest is within a sphere at the origin with radius 0.5
        tf_world_cam_rescaled = frame.tf_cam_world.inverse()
        translation=tf_world_cam_rescaled.translation().copy()
        translation*=SCENE_SCALE
        translation+=SCENE_TRANSLATION
        tf_world_cam_rescaled.set_translation(translation)
        frame.tf_cam_world=tf_world_cam_rescaled.inverse()

        #subsample the image to lower resolution in case you are running on a low VRAM GPU
        frame=frame.subsample(IMG_SUBSAMPLE_FACTOR)

        #append to the scene so the frustums are visualized if the viewer is enabled
        frustum_mesh=frame.create_frustum_mesh(scale_multiplier=0.06)
        Scene.show(frustum_mesh, "frustum_mesh_"+str(idx))

        #finish
        frames.append(frame)
    
    return frames



def run():

    config_file="train_permuto_sdf.cfg"
    config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../../../config', config_file)
    train_params=TrainParams.create(config_path)
    hyperparams=HyperParamsPermutoSDF()


    #get the checkpoints path which will be at the root of the permuto_sdf package 
    permuto_sdf_root=os.path.dirname(os.path.abspath(permuto_sdf.__file__))
    checkpoint_path=os.path.join(permuto_sdf_root, "checkpoints/custom_dataset")
    os.makedirs(checkpoint_path, exist_ok=True)

    
    train_params.set_with_tensorboard(True)
    train_params.set_save_checkpoint(True)
    print("checkpoint_path",checkpoint_path)
    print("with_viewer", with_viewer)

    experiment_name="custom"
    if args.exp_info:
        experiment_name+="_"+args.exp_info
    print("experiment name",experiment_name)


    #CREATE CUSTOM DATASET---------------------------
    frames=create_custom_dataset() 

    #print the scale of the scene which contains all the cameras.
    print("scene centroid", Scene.get_centroid()) #aproximate center of our scene which consists of all frustum of the cameras
    print("scene scale", Scene.get_scale()) #how big the scene is as a measure betwen the min and max of call cameras positions

    ##VISUALIZE
    # view=Viewer.create()
    # while True:
        # view.update()


    ####train
    tensor_reel=MiscDataFuncs.frames2tensors(frames) #make an tensorreel and get rays from all the images at 
    train(args, config_path, hyperparams, train_params, None, experiment_name, with_viewer, checkpoint_path, tensor_reel, frames_train=frames, hardcoded_cam_init=False)



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
