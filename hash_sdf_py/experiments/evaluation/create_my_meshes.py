#!/usr/bin/env python3

#this scripts runs in a batch way through all the checkpoints that we have in list_of_checkpoints and creates meshes out of them

#DTU meshes are in another frame that does not correspond with the ground truth but we want to transform it so that it corresponds
#more info on the evaluation is here
# https://github.com/Totoro97/NeuS/issues/43
#for running Neus you need to run:
# python exp_runner.py --mode validate_mesh --conf ./confs/wmask.conf --case "dtu_scan24" --is_continue
# python exp_runner.py --mode validate_mesh --conf ./confs/womask.conf --case "bvms_bear" --is_continue
#and you need to modify conf/wmask to point to the DTU dataset and the checkpoints path

#For this script the checkpoints are in ./instant_ngp_2_py/scripts_for_paper/list_of_checkpoints.py

###CALL with 
# ./instant_ngp_2_py/scripts_for_paper/create_my_meshes.py --dataset dtu  --res 700 --model_type hashdf   [--use_all_imgs  --without_mask]








import torch

import sys
import os
import numpy as np
from tqdm import tqdm
import time
import torchvision
import argparse

from easypbr  import *
from dataloaders import *
from instant_ngp_2  import TrainParams
from instant_ngp_2  import ModelParams
from instant_ngp_2  import EvalParams
from instant_ngp_2  import NGPGui
from instant_ngp_2  import InstantNGP
from instant_ngp_2  import OccupancyGrid
from instant_ngp_2_py.instant_ngp_2.models import *
from instant_ngp_2_py.instant_ngp_2.models_neus_custom import SDFNetwork
from instant_ngp_2_py.instant_ngp_2.sdf_utils import *
from instant_ngp_2_py.utils.aabb import *
from instant_ngp_2_py.utils.sphere import *
from instant_ngp_2_py.utils.frame_py import *


from instant_ngp_2_py.utils.utils import create_dataloader
from instant_ngp_2_py.utils.utils import create_bb_for_dataset
from instant_ngp_2_py.utils.utils import create_bb_mesh

from skimage import measure

import instant_ngp_2_py.scripts_for_paper.list_of_checkpoints as list_chkpts
import instant_ngp_2_py.scripts_for_paper.list_of_training_scenes as list_scenes

from pyhocon import ConfigFactory




config_file="create_my_meshes.cfg"

torch.manual_seed(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
torch.set_grad_enabled(False)
config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../../config', config_file)

#argparse
parser = argparse.ArgumentParser(description='prepare dtu evaluation')
parser.add_argument('--dataset', required=True,  default="",  help="dataset which can be dtu or bmvs")
parser.add_argument('--res', required=True,  help="Resolution of the mesh, usually t least 700")
parser.add_argument('--restrict_to_scene',  help="If you want only one scene you scan set this to dtu_scene24 for example")
parser.add_argument('--use_all_imgs', action='store_true', help="Use all images instead of just the training set")
parser.add_argument('--without_mask', action='store_true', help="Set this to true in order to train without a mask and model the BG differently")
parser.add_argument('--model_type', required=True, help='Model, can be hashsdf, ingp')
args = parser.parse_args()





#####PARAMETERS#######
dataset=args.dataset
scene_name=args.restrict_to_scene
use_all_imgs = args.use_all_imgs
without_mask = args.without_mask
model_type=args.model_type
config_training="without_mask_"+str(without_mask)+"_use_all_imgs_"+str(use_all_imgs)
nr_points_per_dim=int(args.res) #on cuda3 we can use up to a res of 1900 which uses around 30GB of memory. Res of 2300 uses 49gb
use_home=False 
with_viewer=False
# path_gt_meshes="/media/rosu/Data/data/dtu/Points/stl/"

#path for home or remote
if use_home:
    if model_type=="hashsdf":
        path_for_output_my_meshes=os.path.join("/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/outputs/output_my_meshes/",dataset)
    elif model_type=="ingp":
        path_for_output_my_meshes=os.path.join("/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/outputs/output_ingp_meshes/",dataset)
    elif model_type=="neus":
        path_for_output_my_meshes=os.path.join("/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/outputs/output_neus_meshes/",dataset)

else:
    if model_type=="hashsdf":
        path_for_output_my_meshes=os.path.join("/home/user/rosu/c_ws/src/instant_ngp_2/outputs/output_my_meshes/",dataset)
    elif model_type=="ingp":
        path_for_output_my_meshes=os.path.join("/home/user/rosu/c_ws/src/instant_ngp_2/outputs/output_ingp_meshes/",dataset)
    elif model_type=="neus":
        path_for_output_my_meshes=os.path.join("/home/user/rosu/c_ws/src/instant_ngp_2/outputs/output_neus_meshes/",dataset)


# assert os.path.isdir(path_gt_meshes), "Path for path_gt_meshes does not exist. Maybe you are on the wrong computer?"
assert os.path.isdir(path_for_output_my_meshes), "Path for path_for_output_my_meshes does not exist. Maybe you are on the wrong computer?"


# loader, _= create_dataloader(dataset, scene_name, config_path, use_home, args.use_all_imgs, args.without_mask)



def extract_mesh_and_transform_to_original_tf(model, lattice, nr_points_per_dim, loader, aabb):
    # extracted_mesh=extract_mesh_from_sdf_model(model, lattice, nr_points_per_dim=600, min_val=-0.5, max_val=0.5)

    if isinstance(model, SDF):
        extracted_mesh=extract_mesh_from_sdf_model(model, lattice, nr_points_per_dim=nr_points_per_dim, min_val=-0.5, max_val=0.5)
    elif isinstance(model, INGP):
        extracted_mesh=extract_mesh_from_density_model(model, lattice, nr_points_per_dim=nr_points_per_dim, min_val=-0.5, max_val=0.5, threshold=40)
        # extracted_mesh.flip_normals()
    elif isinstance(model, SDFNetwork):
        extracted_mesh=extract_mesh_from_sdf_model_neus(model, nr_points_per_dim=nr_points_per_dim, min_val=-0.5, max_val=0.5)
        

    extracted_mesh=aabb.remove_points_outside(extracted_mesh)
    extracted_mesh.recalculate_min_max_height()
   
    #transform the extracted mesh from the easypbr coordinate frame to the dtu one so that it matches the gt
    if isinstance(loader, DataLoaderDTU):
        tf_easypbr_dtu=loader.get_tf_easypbr_dtu()
        tf_dtu_easypbr=tf_easypbr_dtu.inverse()
        extracted_mesh.transform_model_matrix(tf_dtu_easypbr.to_double())
        extracted_mesh.apply_model_matrix_to_cpu(True)

    return extracted_mesh



def run():
    if with_viewer:
        view=Viewer.create(config_path)
        ngp_gui=NGPGui.create(view)

    

    #params
    nr_lattice_features=2
    nr_resolutions=24

    #torch stuff 
    lattice=Lattice.create(config_path, "lattice")
    nr_lattice_vertices=lattice.capacity()
    print("nr_lattice_vertices ", nr_lattice_vertices)



    aabb, aabb_big = create_bb_for_dataset(dataset)
    aabb_py=SpherePy(radius=0.5, center=[0,0,0])
    # bb_mesh = create_bb_mesh(aabb) 
    # Scene.show(bb_mesh,"bb_mesh")



    
    # cos_anneal_ratio=1.0
    # model=SDF(nr_lattice_vertices, nr_lattice_features, nr_resolutions=nr_resolutions, boundary_primitive=aabb_big, feat_size_out=32, nr_iters_for_c2f=10000, N_initial_samples=64, N_samples_importance=64, N_iters_upsample=4 ).to("cuda")
    if model_type=="hashsdf":
        cos_anneal_ratio=1.0
        #model 
        model=SDF(nr_lattice_vertices, nr_lattice_features, nr_resolutions=nr_resolutions, boundary_primitive=aabb_big, feat_size_out=32, nr_iters_for_c2f=10000, N_initial_samples=64, N_samples_importance=64, N_iters_upsample=4 ).to("cuda")
    elif model_type=="ingp":
        model=INGP(3, nr_lattice_vertices, nr_lattice_features, nr_resolutions=nr_resolutions, boundary_primitive=aabb, nr_iters_for_c2f=1 ).to("cuda")
        occupancy_grid=OccupancyGrid(256, 1.0, [0,0,0])
    elif model_type=="neus":
        # Configuration
        if use_home:
            conf_path = "/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/config/neus_womask.conf" 
        else:
            conf_path = "/home/user/rosu/c_ws/src/instant_ngp_2/config/neus_womask.conf" 
        f = open(conf_path)
        conf_text = f.read()
        # conf_text = conf_text.replace('CASE_NAME', case)
        f.close()
        conf = ConfigFactory.parse_string(conf_text)
        model = SDFNetwork(**conf['model.sdf_network']).cuda()
    else:
        print("not known model", model_type)
        exit()
    
    # run over every scene 
    # nr_scenes=loader.nr_scenes()

    scenes_list=list_scenes.datasets[dataset]
    
    # for scene_idx in range(len(scenes_list)):
    for scan_name in scenes_list:

        loader, _= create_dataloader(dataset, scan_name, config_path, use_home, args.use_all_imgs, args.without_mask)

        # nr_samples=loader.nr_samples()

        # scan_name=loader.get_current_scene_name()

        print("extracting mesh for scene_name", scan_name)

        #get the list of checkpoints
        scene_config=dataset+"_"+config_training
        if model_type=="ingp":
            scene_config+="_ingp"
        elif model_type=="neus":
            scene_config+="_neus"
        ckpts=list_chkpts.ckpts[scene_config]
        # print("ckpts",ckpts)
        if use_home:
            path_prefix=ckpts["path_prefix_home"]
        else:
            path_prefix=ckpts["path_prefix_remote"]

        ckpt_for_scene=ckpts[scan_name]
        print("ckpt_for_scene",ckpt_for_scene)
        if model_type=="hashsdf":
            checkpoint_path_sdf=os.path.join(path_prefix,ckpt_for_scene,"models/sdf_model.pt")
        elif model_type=="ingp":
            checkpoint_path_sdf=os.path.join(path_prefix,ckpt_for_scene,"models/ingp_model.pt")
            occupancy_grid.set_grid_values(torch.load( os.path.join(path_prefix,ckpt_for_scene,"models/grid_values.pt") ))
            occupancy_grid.set_grid_occupancy(torch.load( os.path.join(path_prefix,ckpt_for_scene,"models/grid_occupancy.pt")  ))
        elif model_type=="neus":
            checkpoint_path_sdf=os.path.join(path_prefix,ckpt_for_scene,"models/neus_sdf_model.pt")

        print("checkpoint_path_sdf", checkpoint_path_sdf)


        #load sdf model
        model.load_state_dict(torch.load(checkpoint_path_sdf) )
        model.eval()


        #extract my mesh
        extracted_mesh=extract_mesh_and_transform_to_original_tf(model, lattice, nr_points_per_dim, loader, aabb_py)
        if with_viewer:
            Scene.show(extracted_mesh,"extracted_mesh"+scan_name)

        #remove points that are in unoccpied space
        #causes weird holes
        # vertices=torch.from_numpy(extracted_mesh.V.copy()).cuda().float()
        # occupancy_val=occupancy_grid.check_occupancy(vertices) #true for occupied, false for empty
        # extracted_mesh.remove_marked_vertices( occupancy_val.flatten().bool().cpu().numpy() ,True)


        # #write my mesh
        os.makedirs(os.path.join(path_for_output_my_meshes,config_training), exist_ok=True)
        path_output=os.path.join(path_for_output_my_meshes, config_training, scan_name+".ply")
        extracted_mesh.save_to_file(path_output)



        # # #show gt mesh
        # # if dataset=="dtu":
        # #     nr_of_scene=int(scan_name.replace('dtu_scan', ''))
        # #     nr_of_scene_leading_zeros=str(nr_of_scene).zfill(3)
        # #     name_of_gt_mesh="stl"+nr_of_scene_leading_zeros+"_total.ply"
        # #     path_gt_mesh=os.path.join(path_gt_meshes,name_of_gt_mesh)
        # #     gt_mesh=Mesh(path_gt_mesh)
        # #     Scene.show(gt_mesh,"gt_mesh")

   
        # loader.start_reading_next_scene()

        # break

    


    if with_viewer:
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