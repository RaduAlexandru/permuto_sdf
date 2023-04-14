import os
import torch
from typing import Optional
import math
import numpy as np
from collections import namedtuple
import torch.nn.functional as F



from permuto_sdf_py.utils.nerf_utils import get_midpoint_of_sections
from permuto_sdf_py.utils.sdf_utils import sdf_loss_spheres
from permuto_sdf_py.utils.sdf_utils import sdf_loss_sphere
from permuto_sdf  import OccupancyGrid
from permuto_sdf  import VolumeRendering
from permuto_sdf  import RaySamplesPacked
from permuto_sdf  import RaySampler
from easypbr import Camera #for get_frames_cropped



def module_exists(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True



def init_losses():
    loss=0.0
    loss_rgb=0.0
    loss_eikonal=0.0
    loss_curvature=0.0
    loss_lipshitz=0.0

    return loss, loss_rgb, loss_eikonal, loss_curvature, loss_lipshitz



def rgb_loss(gt_rgb, pred_rgb, does_ray_intersect_primitive):
    loss_rgb_l1= ((gt_rgb - pred_rgb).abs()*does_ray_intersect_primitive*1.0 ).mean()
    # epsilon_charbonier=0.001
    # loss_rgb_charbonier= (  torch.sqrt((gt_rgb - pred_rgb)**2 + epsilon_charbonier*epsilon_charbonier)   *does_ray_intersect_primitive*1.0 ) #Charbonnier loss from mipnerf360, acts like l2 when error is small and like l1 when error is large
    return loss_rgb_l1.mean()

def eikonal_loss(sdf_gradients):
    gradient_error = (torch.linalg.norm(sdf_gradients.reshape(-1, 3), ord=2, dim=-1) - 1.0) ** 2
    return gradient_error.mean()

def loss_sphere_init(dataset_name, nr_points, aabb, model,  iter_nr_for_anneal ):
    offsurface_points=aabb.rand_points_inside(nr_points=nr_points)
    offsurface_sdf, offsurface_sdf_gradients, feat = model.get_sdf_and_gradient(offsurface_points, iter_nr_for_anneal)
    #for phenorob
    if dataset_name=="phenorobcp1":
        sphere_ground=SpherePy(radius=2.0, center=[0,-2.4,0])
        sphere_plant=SpherePy(radius=0.15, center=[0,0,0])
        spheres=[sphere_ground, sphere_plant]
        # spheres=[sphere_ground]
        loss, loss_sdf, gradient_error=sdf_loss_spheres(offsurface_points, offsurface_sdf, offsurface_sdf_gradients, spheres, distance_scale=1.0)
    elif dataset_name=="bmvs":
        loss, loss_sdf, gradient_error=sdf_loss_sphere(offsurface_points, offsurface_sdf, offsurface_sdf_gradients, sphere_radius=0.3, sphere_center=[0,0,0], distance_scale=1.0)
    elif dataset_name=="dtu":
        loss, loss_sdf, gradient_error=sdf_loss_sphere(offsurface_points, offsurface_sdf, offsurface_sdf_gradients, sphere_radius=0.3, sphere_center=[0,0,0], distance_scale=1.0)
    elif dataset_name=="easypbr":
        loss, loss_sdf, gradient_error=sdf_loss_sphere(offsurface_points, offsurface_sdf, offsurface_sdf_gradients, sphere_radius=0.3, sphere_center=[0,0,0], distance_scale=1.0)
    elif dataset_name=="multiface":
        loss, loss_sdf, gradient_error=sdf_loss_sphere(offsurface_points, offsurface_sdf, offsurface_sdf_gradients, sphere_radius=0.3, sphere_center=[0,0,0], distance_scale=1.0)
    else:
        # print("Using default sphere loss")
        loss, loss_sdf, gradient_error=sdf_loss_sphere(offsurface_points, offsurface_sdf, offsurface_sdf_gradients, sphere_radius=0.3, sphere_center=[0,0,0], distance_scale=1.0)
        # print("dataset not known")
        # exit()

    return loss, loss_sdf, gradient_error


def get_iter_for_anneal(iter_nr, nr_iter_sphere_fit):
    if iter_nr<nr_iter_sphere_fit:
        #####DO NOT DO THIS , it sets the iterfor anneal so high that is triggers the decay of the lipthiz loss to 0
        # iter_nr_for_anneal=999999 #dont do any c2f when fittign sphere 
        iter_nr_for_anneal=iter_nr 
    else:
        iter_nr_for_anneal=iter_nr-nr_iter_sphere_fit

    return iter_nr_for_anneal


class CropStruct():
    def __init__(self, start_x, start_y, crop_width, crop_height):
        self.start_x = start_x
        self.start_y = start_y
        self.crop_width = crop_width
        self.crop_height = crop_height
        

def get_frames_cropped(loader_train, aabb):
    frames_train=[]
    max_width=0
    max_height=0
    # CropStruct = namedtuple("CropStruct", "start_x start_y crop_width crop_height")
    list_of_true_crops=[]
    for i in range(loader_train.nr_samples()):
        frame=loader_train.get_frame_at_idx(i)
        if frame.is_shell:
            frame.load_images()
        cam=Camera()
        coords_2d_center=frame.project([0,0,0]) #project the center (the dataloader already ensures that the plant in interest is at the center)
        #project also along the x and y axis of the camera to get the bounds of the sphere in camera coords
        cam.from_frame(frame)
        cam_axes=cam.cam_axes()
        x_axis=cam_axes[:,0]
        y_axis=cam_axes[:,1]
        radius=aabb.m_radius
        coords_2d_x_positive=frame.project(x_axis*radius)
        coords_2d_x_negative=frame.project(-x_axis*radius)
        coords_2d_y_positive=frame.project(y_axis*radius)
        coords_2d_y_negative=frame.project(-y_axis*radius)
        #for cropping
        start_x=int(coords_2d_x_negative[0])
        start_y=int(coords_2d_y_positive[1])
        width=int(coords_2d_x_positive[0]-coords_2d_x_negative[0])
        height=int(coords_2d_y_negative[1]-coords_2d_y_positive[1])
        print("start_x",start_x)
        print("start_y",start_y)
        print("width",width)
        print("height",height)
        start_x, start_y, width, height=frame.get_valid_crop(start_x, start_y, width, height)
        # frame=frame.crop(start_x, start_y, width, height, True)
        # frames_train.append(frame)
        # Gui.show(frame.rgb_32f, " frame "+str(i) ) 
        if width>max_width:
            max_width=width
        if height>max_height:
            max_height=height
        true_crop = CropStruct(start_x, start_y, width, height)
        list_of_true_crops.append(true_crop)

    #iterate again to set the maximum width and height to whatever the smallest frame is, because we don;t want max_width to be bigger than any of the frames
    for i in range(loader_train.nr_samples()):
        frame=loader_train.get_frame_at_idx(i)
        if frame.is_shell:
            frame.load_images()
        if max_width>frame.width-1:
            max_width=frame.width-1
        if max_height>frame.height-1:
            max_height=frame.height-1

    #adjust the true crops so that none of them crops more than max_width or max_height
    for i in range(loader_train.nr_samples()):
        true_crop=list_of_true_crops[i]
        if true_crop.crop_width>max_width:
            true_crop.crop_width=max_width
        if true_crop.crop_height>max_height:
            true_crop.crop_height=max_height
    
    print("max_width",max_width)
    print("max_height",max_height)
    #upsampel the true crops so that they are still inside the image but they all have the same width and height so that we can use the tensorreel
    frames_train_cropped_equal=[]
    for i in range(loader_train.nr_samples()):
        frame=loader_train.get_frame_at_idx(i)
        if frame.is_shell:
            frame.load_images()
        true_crop=list_of_true_crops[i]
        start_x, start_y, width, height=frame.enlarge_crop_to_size(true_crop.start_x, true_crop.start_y, true_crop.crop_width, true_crop.crop_height, max_width, max_height) #crops then all to the same size
        # print("enlarged crop has x,y, width and height ", start_x, start_y, width, height)
        # print("will now crop a frame of size width and height ", frame.width, frame.height)
        # print("true crop", true_crop)
        frame=frame.crop(start_x, start_y, width, height, True)
        loader_train.get_frame_at_idx(i).unload_images()#now that we cropped to what we want we can unload the frame
        print("frame shouls all have equal size now ", frame.width, " ", frame.height)
        frames_train_cropped_equal.append(frame)
        # Gui.show(frame.rgb_32f, " frame "+str(i) ) 
    frames_train=frames_train_cropped_equal

    return frames_train



def color_by_idx(nr_voxels):
    #color by idx
    colors_t=torch.range(0,nr_voxels-1)
    #repeat and assign color
    colors_t=colors_t.view(-1,1).repeat(1,3)
    colors_t=colors_t.float()/nr_voxels

    return colors_t

def color_by_density_from_occupancy_grid(occupancy_grid):
    #color by grid value
    colors_t=occupancy_grid.get_grid_values()
    colors_t=torch.clamp(colors_t,max=1.0)
    #repeat and assign color
    colors_t=colors_t.view(-1,1).repeat(1,3)
    colors_t=colors_t.float()

    return colors_t

def color_by_occupancy_from_occupancy_grid(occupancy_grid):
    #color by grid value
    colors_t=occupancy_grid.get_grid_occupancy().float()
    colors_t=torch.clamp(colors_t,max=1.0)
    #repeat and assign color
    colors_t=colors_t.view(-1,1).repeat(1,3)
    colors_t=colors_t.float()

    return colors_t

def color_by_density(density):
    #color by grid value
    colors_t=density
    colors_t=torch.clamp(colors_t,max=1.0)
    #repeat and assign color
    colors_t=colors_t.view(-1,1).repeat(1,3)
    colors_t=colors_t.float()

    return colors_t

def load_from_checkpoint(ckpt_path_full, model_sdf, model_rgb, model_bg, occupancy_grid):

    checkpoint_path_sdf=os.path.join(ckpt_path_full,"sdf_model.pt")
    checkpoint_path_rgb=os.path.join(ckpt_path_full,"rgb_model.pt")
    checkpoint_path_bg=os.path.join(ckpt_path_full,"nerf_hash_model_bg.pt")
    checkpoint_path_grid_values=os.path.join(ckpt_path_full,"grid_values.pt")
    checkpoint_path_grid_occupancy=os.path.join(ckpt_path_full,"grid_occupancy.pt")

    model_sdf.load_state_dict(torch.load(checkpoint_path_sdf) )
    model_rgb.load_state_dict(torch.load(checkpoint_path_rgb) )
    model_bg.load_state_dict(torch.load(checkpoint_path_bg) )
    model_sdf.eval()
    model_rgb.eval()
    model_bg.eval()
    occupancy_grid.set_grid_values(torch.load( checkpoint_path_grid_values ))
    occupancy_grid.set_grid_occupancy(torch.load( checkpoint_path_grid_occupancy  ))
