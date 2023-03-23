#!/usr/bin/env python3

#get chamfer distance comparisons on DTU between gt_mesh, ours
# It assumes we have created our meshes using create_my_meshes.py and they are at PACKAGE_ROOT/results/output_permuto_sdf_meshes/

####Call with######
# ./permuto_sdf_py/experiments/evaluation/evaluate_chamfer_distance.py --comp_name comp_1  [--with_mask]




import torch
import torchvision

import sys
import os
import numpy as np
import time
import argparse

import permuto_sdf
from easypbr  import *
from dataloaders import *
from permuto_sdf  import NGPGui
from permuto_sdf_py.utils.common_utils import create_dataloader
import permuto_sdf_py.paths.list_of_training_scenes as list_scenes

import numpy as np
import cv2 as cv
from glob import glob
from scipy.io import loadmat
import trimesh

import subprocess



config_file="train_permuto_sdf.cfg"

torch.manual_seed(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../../../config', config_file)




class EvalResults:
    def __init__(self):
        self.nr_scenes_evaluated=0
        self.total_d2s=0
        self.total_s2d=0
        self.total_overall=0
        self.scene_nr2overall = {}
    
    def update(self, scene_nr, d2s, s2d, overall):
        self.total_d2s+=d2s
        self.total_s2d+=s2d
        self.total_overall+=overall
        self.nr_scenes_evaluated+=1

        # print("adding overall", overall)
        # print("before adding scene2nroverall is ", self.scene_nr2overall)
        self.scene_nr2overall[scene_nr]=overall

    def get_results_avg(self):
        mean_d2s = self.total_d2s/self.nr_scenes_evaluated
        mean_s2d = self.total_s2d/self.nr_scenes_evaluated
        mean_overall = self.total_overall/self.nr_scenes_evaluated

        return mean_d2s, mean_s2d, mean_overall

    def get_results_per_scene(self):
        # print("returningdict ", self.scene_nr2overall)
        # print("in this object the total overall is ", self.total_overall)
        # print("in this object the nr_scenes is ", self.nr_scenes_evaluated)

        return self.scene_nr2overall 

def run_eval(mesh_path, scene_nr, dataset_dir, output_path ):
    
    #get the path to the evaluation script which is in ./DTUeval-python/eval.py
    cur_file_path=os.path.dirname(os.path.abspath(__file__))
    eval_script_path=os.path.join(cur_file_path,"DTUeval-python/eval.py")

    
    output_eval=subprocess.check_output([
            "python3", 
            eval_script_path, 
            "--data", mesh_path, 
            "--scan", scene_nr, 
            "--mode", "mesh", 
            "--dataset_dir", dataset_dir, 
            "--vis_out_dir", output_path
            ],  shell=False)

    #parse the outputs in the 3 components of mean_d2s, mean_s2d, over_all
    #the format is b'1.031434859827024 1.5483653154552615 1.2899000876411426\n'

    #remove letters and split
    output_eval=output_eval.decode("utf-8")
    output_eval=output_eval.split()
    output_eval=[float(x) for x in output_eval] #get to list fo floats


    return output_eval


#meshes trained without mask are actually cleaned by the mask by this code 
# https://github.com/Totoro97/NeuS/issues/74
def clean_points_by_mask(points, scan, loader):
    dataset_path=loader.get_dataset_path()
    cameras = np.load( os.path.join(dataset_path, 'dtu_scan{}/cameras_sphere.npz'.format(scan))  )
    mask_lis = sorted(glob(   os.path.join(dataset_path, 'dtu_scan{}/mask/*.png'.format(scan))   ))
    # if use_home:
        # cameras = np.load('/media/rosu/Data/data/neus_data/data_DTU/dtu_scan{}/cameras_sphere.npz'.format(scan))
        # mask_lis = sorted(glob('/media/rosu/Data/data/neus_data/data_DTU/dtu_scan{}/mask/*.png'.format(scan)))
    # else:
        # cameras = np.load('/home/user/rosu/data/neus_data/data_DTU/dtu_scan{}/cameras_sphere.npz'.format(scan))
        # mask_lis = sorted(glob('/home/user/rosu/data/neus_data/data_DTU/dtu_scan{}/mask/*.png'.format(scan)))
    n_images = 49 if scan < 83 else 64
    inside_mask = np.ones(len(points)) > 0.5
    for i in range(n_images):
        P = cameras['world_mat_{}'.format(i)]
        pts_image = np.matmul(P[None, :3, :3], points[:, :, None]).squeeze() + P[None, :3, 3]
        pts_image = pts_image / pts_image[:, 2:]
        pts_image = np.round(pts_image).astype(np.int32) + 1

        mask_image = cv.imread(mask_lis[i])
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (101, 101))
        mask_image = cv.dilate(mask_image, kernel, iterations=1)
        mask_image = (mask_image[:, :, 0] > 128)

        mask_image = np.concatenate([np.ones([1, 1600]), mask_image, np.ones([1, 1600])], axis=0)
        mask_image = np.concatenate([np.ones([1202, 1]), mask_image, np.ones([1202, 1])], axis=1)

        curr_mask = mask_image[(pts_image[:, 1].clip(0, 1201), pts_image[:, 0].clip(0, 1601))]

        inside_mask &= curr_mask.astype(bool)

    return inside_mask
def clean_mesh(old_mesh_path, scan, loader):
    old_mesh = trimesh.load(old_mesh_path)
    old_vertices = old_mesh.vertices[:]
    old_faces = old_mesh.faces[:]
    mask = clean_points_by_mask(old_vertices, scan, loader)
    indexes = np.ones(len(old_vertices)) * -1
    indexes = indexes.astype(np.int64)
    indexes[np.where(mask)] = np.arange(len(np.where(mask)[0]))

    faces_mask = mask[old_faces[:, 0]] & mask[old_faces[:, 1]] & mask[old_faces[:, 2]]
    new_faces = old_faces[np.where(faces_mask)]
    new_faces[:, 0] = indexes[new_faces[:, 0]]
    new_faces[:, 1] = indexes[new_faces[:, 1]]
    new_faces[:, 2] = indexes[new_faces[:, 2]]
    new_vertices = old_vertices[np.where(mask)]

    new_mesh = trimesh.Trimesh(new_vertices, new_faces)
    
    meshes = new_mesh.split(only_watertight=False)
    new_mesh = meshes[np.argmax([len(mesh.faces) for mesh in meshes])]

    #make an easypbr mesh
    new_mesh_easypbr=Mesh()
    new_mesh_easypbr.V=new_mesh.vertices
    new_mesh_easypbr.F=new_mesh.faces

    return new_mesh_easypbr


def run():

    dataset="dtu"
    low_res=False
    with_viewer=False

    if with_viewer:
        view=Viewer.create(config_path)
        ngp_gui=NGPGui.create(view)

    #argparse
    parser = argparse.ArgumentParser(description='Quantitative comparison')
    parser.add_argument('--comp_name', required=True,  help='Tells which computer are we using which influences the paths for finding the data') 
    parser.add_argument('--with_mask', action='store_true', help="Set this to true in order to train with a mask")
    args = parser.parse_args()

    config_training="with_mask_"+str(args.with_mask)
    # if not args.with_mask:
        # config_training+="_clean" #we use the cleaned up mesh that we get after runnign clean_mesh.py

    #path of my meshes
    permuto_sdf_root=os.path.dirname(os.path.abspath(permuto_sdf.__file__))
    results_path=os.path.join(permuto_sdf_root, "results")
    my_meshes_path=os.path.join(results_path,"output_permuto_sdf_meshes")
    #path for gt
    gt_meshes_path="/media/rosu/Data/data/dtu/data_prepared_for_gt"
    #outputs path where to dump the results after the evaluation
    output_path=os.path.join(results_path,"output_eval_chamfer_dist",dataset,config_training)
    os.makedirs( output_path, exist_ok=True)




    first_time=True

    results_mine=EvalResults() 

    scenes_list=list_scenes.datasets[dataset]


    for scan_name in scenes_list:
        print("scan_name",scan_name)
        loader_train, loader_test= create_dataloader(config_path, dataset, scan_name, low_res, args.comp_name, args.with_mask) 
        loader=loader_train
        nr_samples=loader.nr_samples()
        #get the scan_nr
        scan_nr=int(scan_name.replace('dtu_scan', ''))


        #get my mesh for this scene
        my_mesh_path=os.path.join(my_meshes_path, "dtu", config_training, scan_name+".ply")
        print("my_mesh_path is ", my_mesh_path)
        if not os.path.isfile(my_mesh_path):
            print("######COULD NOT FIND:  ", my_mesh_path)
            exit(1)
        else:
            #my mesh
            my_mesh=Mesh(my_mesh_path)
            #clean my mesh if it was obtained without a mask supervision because NeuS does the same
            if not args.with_mask:
                print("cleaning mesh")
                my_mesh=clean_mesh(my_mesh_path, scan_nr, loader)
                #write mesh to a temporary file because the eval script expect to read something from a file
                tmp_mesh_path=os.path.join(results_path,"tmp")
                os.makedirs( tmp_mesh_path, exist_ok=True)
                tmp_mesh_save_path=os.path.join(tmp_mesh_path,"tmp.ply")
                my_mesh.save_to_file(tmp_mesh_save_path)
                my_mesh_path=tmp_mesh_save_path
                print("tmp_my_mesh_path",my_mesh_path)



            #get the gt mesh
            scene_name=loader.get_current_scene_name()
            scene_nr=scene_name.replace('dtu_scan', '')
            scene_nr_filled=scene_nr.zfill(3)
            print("scene_nr_filled",scene_nr_filled)
            gt_mesh_name="stl"+scene_nr_filled+"_total.ply"
            gt_mesh_path=os.path.join(gt_meshes_path,"Points/stl",gt_mesh_name)
            gt_mesh=Mesh(gt_mesh_path)


            if with_viewer:
                Scene.show(my_mesh,"my_mesh")
                Scene.show(gt_mesh,"gt_mesh")
                # while True:
                view.update()


            #run DTU evaluation 
            # https://github.com/jzhangbs/DTUeval-python
            output_eval_my_mesh=run_eval(my_mesh_path, scene_nr, gt_meshes_path, output_path)
            print("output_eval_my_mesh", output_eval_my_mesh)
            results_mine.update(scene_nr, output_eval_my_mesh[0], output_eval_my_mesh[1], output_eval_my_mesh[2])


      

    #finished reading all scenes
    #print results
    ####MINE
    print("---------MINE--------")
    mine_avg=results_mine.get_results_avg() 
    mine_per_scene=results_mine.get_results_per_scene()
    print("mine_avg_overall", mine_avg[2])
    print("mine_per_scene", mine_per_scene)
    


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