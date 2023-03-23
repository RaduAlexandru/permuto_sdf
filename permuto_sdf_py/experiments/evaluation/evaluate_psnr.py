#!/usr/bin/env python3

#calculates psnr for the task of novel view synthesis. 
# It assumes we have created our images using create_my_images.py and they are at PACKAGE_ROOT/results/output_permuto_sdf_images/
#we can only evaluate the models that were trained without mask supervision because neus only provides those results

####Call with######
# ./permuto_sdf_py/experiments/evaluation/evaluate_psnr.py --comp_name comp_1 




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


from piq import psnr, ssim
import piq

import subprocess




config_file="train_permuto_sdf.cfg"

torch.manual_seed(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../../../config', config_file)




#stores the results for a certain scene
class EvalResultsPerScene:
    def __init__(self):
        self.total_psnr=0
        self.total_ssim=0
        self.total_lpips=0
        self.nr_imgs_evaluated=0
    
    def update(self, psnr, ssim, lpips):
        self.total_psnr+=psnr
        self.total_ssim+=ssim
        self.total_lpips+=lpips
        self.nr_imgs_evaluated+=1

    def results_averaged(self):
        psnr=self.psnr_avg()
        ssim=self.ssim_avg()
        lpips=self.lpips_avg()

        return psnr, ssim, lpips

    def psnr_avg(self):
        psnr=self.total_psnr/self.nr_imgs_evaluated
        return psnr

    def ssim_avg(self):
        ssim=self.total_ssim/self.nr_imgs_evaluated
        return ssim

    def lpips_avg(self):
        lpips=self.total_lpips/self.nr_imgs_evaluated
        return lpips



class EvalResults:
    def __init__(self,name):
        self.scenes={}
        self.name=name

    def scene_name_to_nr(self, scene_name):
        scene_name_nr=scene_name.replace('dtu_scan', '')
        return int(scene_name_nr)
    
    def update(self, scene_name, psnr, ssim, lpips):
        scene_name=self.scene_name_to_nr(scene_name)

        if scene_name not in self.scenes:
            self.scenes[scene_name]=EvalResultsPerScene()

        self.scenes[scene_name].update(psnr,ssim,lpips)

    def get_results_for_scene(self,scene_name):
        scene_name=self.scene_name_to_nr(scene_name)

        return  self.scenes[scene_name].results_averaged()

    def get_results_for_scene_string(self,scene_name):
        scene_name=self.scene_name_to_nr(scene_name)

        psnr, ssim, lpips=self.scenes[scene_name].results_averaged()
        s="psnr: " + str(psnr) + " ssim: " + str(ssim) + " lpips: " + str(lpips) 
        return s

    def print_results_for_all_scenes(self):
        # return  self.scenes[scene_name].results_averaged()
        scenes_sorted={k: v for k, v in sorted(self.scenes.items(), key=lambda item: item[0])}
        scenes_string=" "
        psnr_string=" "
        ssim_string=" "

        #get also the avg psnr and avg ssim
        nr_scenes=0
        psnr_total=0
        ssim_total=0

        for scene_name, scene_results in scenes_sorted.items():
            scenes_string+=str(scene_name) + " "
            # psnr_string+=str(scene_results.psnr_avg())+" & "
            psnr_string+=str( "{:2.2f}".format(scene_results.psnr_avg()) )+" & "
            ssim_string+=str( "{:2.3f}".format(scene_results.ssim_avg()))+" & "

            #get the avg 
            nr_scenes+=1
            psnr_total+=scene_results.psnr_avg()
            ssim_total+=scene_results.ssim_avg()

            #print again here
            print("scene_name", scene_name, " psnr:", str( "{:2.2f}".format(scene_results.psnr_avg()) ) )

        print("scenes_string", scenes_string)
        print("psnr: ", self.name, " ", psnr_string)
        print("ssim: ", self.name, " ", ssim_string)
        print("psnr_avg ", self.name, " ", psnr_total/nr_scenes)
        print("ssim_avg ", self.name, " ", ssim_total/nr_scenes)


    


def run():

    #argparse
    parser = argparse.ArgumentParser(description='Quantitative comparison')
    parser.add_argument('--comp_name', required=True,  help='Tells which computer are we using which influences the paths for finding the data') 
    args = parser.parse_args()

    first_time=True

    #params
    dataset="dtu"
    with_mask=True #we can only evaluate those that were trained wo mask because that is what neus evalutes for on table 4 and what the author provides images for. But with this we load the images with the mask so we can evaluate only the foreground
    low_res=False
    with_viewer=False


    if with_viewer: 
        view=Viewer.create(config_path)
        ngp_gui=NGPGui.create(view)

    #path of my images
    permuto_sdf_root=os.path.dirname(os.path.abspath(permuto_sdf.__file__))
    results_path=os.path.join(permuto_sdf_root, "results")
    my_imgs_path=os.path.join(results_path,"output_permuto_sdf_images")


    #get for each scene
    # nr_scenes=loader_test.nr_scenes()
    # print("nr_scenes",nr_scenes)

    results_mine=EvalResults("mine") 

    scenes_list=list_scenes.datasets[dataset]


    for scan_name in scenes_list:
        loader_train, loader_test= create_dataloader(config_path, dataset, scan_name, low_res, args.comp_name, with_mask)
        nr_samples=loader_test.nr_samples()
        print("scan_name",scan_name)

        # if scan_name!="dtu_scan63":
        #     loader_test.start_reading_next_scene()
        #     continue


        for i in range(nr_samples):
            frame=loader_test.get_frame_at_idx(i) 
            print("img", frame.frame_idx)

            if frame.is_shell:
                    frame.load_images()

            #get mask
            mask=frame.mask
            mask_tensor=mat2tensor(mask, True)
            #get gt
            gt_img_tensor=mat2tensor(frame.rgb_8u, True).float()/255
            gt_img_tensor=gt_img_tensor*mask_tensor
                

            #load my img
            mine_img_path=os.path.join(my_imgs_path,"dtu/with_mask_False/test",scan_name,"rgb",str(frame.frame_idx)+".png")
            # print("mine_img_path",mine_img_path)


            #mask out the background
            mine_img_mat=Mat(mine_img_path)
            mine_img_tensor=mat2tensor(mine_img_mat, True)/255
            mine_img_tensor=mine_img_tensor[:, 0:3, :, :]
            mine_img_tensor=mine_img_tensor*mask_tensor




            #evaluate mine
            psnr_mine=psnr(mine_img_tensor, gt_img_tensor, data_range=1.).item()
            ssim_mine=ssim(mine_img_tensor, gt_img_tensor, data_range=1.).item()
            lpips_mine=0.0


            results_mine.update(scan_name, psnr_mine, ssim_mine, lpips_mine)

            frame.unload_images()

            #debug imgs
            if with_viewer:
                Gui.show(tensor2mat(mine_img_tensor).rgb2bgr(),"mine_img_tensor")

                view.update()
    

        #finsihed this scene, show the results
        print("results for scene ", scan_name )
        print("mine", results_mine.get_results_for_scene_string(scan_name))

        

    #finished reading all scenes
    #print results
    ####MINE
    results_mine.print_results_for_all_scenes()
    

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