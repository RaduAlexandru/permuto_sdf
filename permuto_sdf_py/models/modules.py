import torch
from torch.autograd import Function
from torch import Tensor
from torch.nn import functional as F
import torchvision

import sys
from permuto_sdf  import PermutoSDF
import numpy as np
import time
import math
import random


#returns a 1D vector of the pixels in a patch that we pick from the image. We assume that the image is flattened into a 1D array. We assume the flattening is done row by row so index 0 is the top left pixel. Can also return None if we choose to sample the full image densely
class PatchPixelPicker(torch.nn.Module):
    def __init__(self ):  
        super(PatchPixelPicker, self).__init__()

    def forward(self, frame, patch_size, pick_all_pixels):

        assert patch_size%2!=0, "The patch size has to be odd"

        #pick random indices
        if pick_all_pixels:
            return None
        else:

            #we create an 1D image of only the indices of each pixels of the image
            nr_pixels=frame.width*frame.height
            full_indices=torch.arange(nr_pixels).view(1,1,frame.height, frame.width)


            #we pick a center pixel such that it has enough pixels up and down 
            half_patch=int(patch_size/2)
            x_idx=random.randint(0,frame.width-patch_size)
            y_idx=random.randint(0,frame.height-patch_size)

            indices_selected=torchvision.transforms.functional.crop(full_indices, y_idx, x_idx, patch_size, patch_size)

            indices_selected=indices_selected.flatten()

            return indices_selected

#same as PatchPixelPicker but instead of the indices corresponding to just one patch, it corresponds to various patches
class PatchesPixelPicker(torch.nn.Module):
    def __init__(self ):  
        super(PatchesPixelPicker, self).__init__()

    def forward(self, frame, patch_size, nr_patches, pick_all_pixels):

        assert patch_size%2!=0, "The patch size has to be odd"


        


        #pick random indices
        if pick_all_pixels:
            return None
        else:

            #we create an 1D image of only the indices of each pixels of the image
            nr_pixels=frame.width*frame.height
            full_indices=torch.arange(nr_pixels).view(1,1,frame.height, frame.width)

            

            #we pick a center pixel such that it has enough pixels up and down 
            half_patch=int(patch_size/2)
            all_indices_list=[]
            for i in range(nr_patches):

                x_idx=random.randint(0,frame.width-patch_size)
                y_idx=random.randint(0,frame.height-patch_size)

                indices_selected=torchvision.transforms.functional.crop(full_indices, y_idx, x_idx, patch_size, patch_size)

                indices_selected=indices_selected.flatten()
                all_indices_list.append(indices_selected)
            all_indices=torch.cat(all_indices_list,0)

            return all_indices

#concatenation of a patch together with some random rays. the random rays help with faster convergence
class PatchAndRandPixelPicker(torch.nn.Module):
    def __init__(self, low_discrepancy ):  
        super(PatchAndRandPixelPicker, self).__init__()

        self.rand_picker=RandPixelPicker(low_discrepancy)
        self.patch_picker=PatchPixelPicker()

    def forward(self, frame, patch_size, nr_random_samples, pick_all_pixels):

        #pick random indices
        if pick_all_pixels:
            return None
        else:
            indices_patch=self.patch_picker(frame, patch_size, pick_all_pixels)
            indices_rand=self.rand_picker(frame, nr_random_samples, pick_all_pixels)

            full_indices=torch.cat([indices_patch, indices_rand], 0)

            return full_indices, indices_patch, indices_rand

#samples for each frame the indices that caused the most error in this image
class ErrorPixelPicker(torch.nn.Module):
    def __init__(self, nr_error_sampled_pixels):  
        super(ErrorPixelPicker, self).__init__()

        self.nr_error_sampled_pixels=nr_error_sampled_pixels

        self.max_errors_dict={} #key is the frame idx and the value is the a tensor of nr_error_sampled_pixels  of the maximum errors that happened for that image 
        self.indices_of_max_errors_dict={} #key is the frame idx and the value is the a tensor of nr_error_sampled_pixels of indices that caused the maximum errors that happened for that image 

    def forward(self, frame):

        if frame.frame_idx not in self.indices_of_max_errors_dict:
            return None 
        else:
            return self.indices_of_max_errors_dict[frame.frame_idx]



    def update(self, frame, indices, errors):

        # print("errors", errors.shape)

        #start the max_errors and indices_of_max_erros dict
        if frame.frame_idx not in self.max_errors_dict:
            self.max_errors_dict[frame.frame_idx]=torch.zeros([self.nr_error_sampled_pixels])
            self.indices_of_max_errors_dict[frame.frame_idx]=torch.zeros([self.nr_error_sampled_pixels]).long()
            print("instatiating a new error map for frame_idx", frame.frame_idx)


        indices_flat=indices.flatten()
        errors_flat=errors.flatten()

        
        #if we store here the same indices that we sampled, update their error
        old_indices=self.indices_of_max_errors_dict[frame.frame_idx]
        old_errors=self.max_errors_dict[frame.frame_idx]
        self.max_errors_dict[frame.frame_idx] = InstantNGP.update_errors_of_matching_indices(old_indices, old_errors, indices_flat, errors_flat)
        #resort the errors
        self.max_errors_dict[frame.frame_idx],_ = torch.sort(self.max_errors_dict[frame.frame_idx], descending=True)


        errors_sorted, indices_that_sort_errors= torch.sort(errors_flat, descending=True)
        indices_sorted=indices_flat[indices_that_sort_errors]

        #get the first worse offendors nr_error_sampled_pixels
        errors_sorted=errors_sorted[0:self.nr_error_sampled_pixels]
        indices_sorted=indices_sorted[0:self.nr_error_sampled_pixels]


        #update the max errors and the indices of those max errors
        max_errors_suparsed=errors_sorted>self.max_errors_dict[frame.frame_idx]
        # print("max_errors_suparsed", max_errors_suparsed)
        self.indices_of_max_errors_dict[frame.frame_idx][max_errors_suparsed]= indices_sorted[max_errors_suparsed]
        self.max_errors_dict[frame.frame_idx][max_errors_suparsed]= errors_sorted[max_errors_suparsed]



#creates ray_origins and ray_dirs of sizes Nx3 given a frame
class CreateRaysModule(torch.nn.Module):
    def __init__(self, precompute_grid=False):
        super(CreateRaysModule, self).__init__()

        #precomputed things
        # self.first_time=True
        # self.width=None
        # self.height=None
        # self.probabilities=None #will get created on the first iteration and stays the same
        self.grid_dict={} #key is the nr of pixels, the val is the probabilities for choosing a pixel for a frame of that size

        self.precompute_grid=precompute_grid

    def compute_grid(self, frame):
        # print("adding CreateRaysModule grid for ", size)
        x_coord= torch.arange(frame.width).view(-1, 1, 1).repeat(1,frame.height, 1)+0.5 #width x height x 1
        y_coord= torch.arange(frame.height).view(1, -1, 1).repeat(frame.width, 1, 1)+0.5 #width x height x 1
        ones=torch.ones(frame.width, frame.height).view(frame.width, frame.height, 1)
        points_2D=torch.cat([x_coord, y_coord, ones],2).transpose(0,1).reshape(-1,3).cuda() #Nx3 we tranpose because we want x cooridnate to be inner most so that we traverse row-wise the image

        return points_2D


    def forward(self, frame, rand_indices):

        if len(self.grid_dict)>50:
            print("We have a list of grid_dict of ", len(self.grid_dict), " and this uses quite some memory. If you are sure this not an issue please ignore")


        if self.precompute_grid:
            #if we don't have probabilities for this size of frame, we add it
            size=(frame.width, frame.height)
            if size not in self.grid_dict:
                print("adding CreateRaysModule grid for ", size)
                self.grid_dict[size]= self.compute_grid(frame)
            points_2D=self.grid_dict[size]
        else:
            #compute the grid
            points_2D=self.compute_grid(frame)


        #get 2d points
        selected_points_2D=points_2D
        if rand_indices!=None:
            selected_points_2D=torch.index_select( points_2D, dim=0, index=rand_indices) 



        #create points in 3D
        K_inv=torch.from_numpy( np.linalg.inv(frame.K) ).to("cuda").float()
        #get from screen to cam coords
        pixels_selected_screen_coords_t=selected_points_2D.transpose(0,1) #3xN
        pixels_selected_cam_coords=torch.matmul(K_inv,pixels_selected_screen_coords_t).transpose(0,1)

        nr_rays=pixels_selected_cam_coords.shape[0]
        pixels_selected_cam_coords=pixels_selected_cam_coords.view(nr_rays, 3)


        #get from cam_coords to world_coords
        tf_world_cam=frame.tf_cam_world.inverse()
        R=torch.from_numpy( tf_world_cam.linear().copy() ).to("cuda").float()
        t=torch.from_numpy( tf_world_cam.translation().copy() ).to("cuda").view(1,3).float()
        pixels_selected_world_coords=torch.matmul(R, pixels_selected_cam_coords.transpose(0,1).contiguous() ).transpose(0,1).contiguous()  + t
        #get direction
        ray_dirs = pixels_selected_world_coords-t
        ray_dirs=F.normalize(ray_dirs, p=2, dim=1)

   
        #ray_origins
        ray_origins=t.repeat(nr_rays,1)

        

        return ray_origins, ray_dirs


class PositionalEncoding(torch.nn.Module):
    def __init__(self, in_channels, num_encoding_functions, only_sin ):
        super(PositionalEncoding, self).__init__()
        self.in_channels=in_channels
        self.num_encoding_functions=num_encoding_functions
        self.only_sin=only_sin

        out_channels=in_channels*self.num_encoding_functions*2

        self.conv= torch.nn.Linear(in_channels, int(out_channels/2), bias=False).cuda()  #in the case we set the weight ourselves
        self.init_weights()


        #we dont train because that causes it to overfit to the input views and not generalize the specular effects to novel views
        self.conv.weight.requires_grad = False

    def init_weights(self):
        with torch.no_grad():
            num_input = self.in_channels
            self.conv.weight.uniform_(-np.sqrt(6 / num_input) , np.sqrt(6 / num_input) )
            # print("weight is ", self.conv.weight.shape) #60x3

            #we make the same as the positonal encoding, which is mutiplying each coordinate with this linespaced frequencies
            lin=2.0 ** torch.linspace(
                0.0,
                self.num_encoding_functions - 1,
                self.num_encoding_functions,
                dtype=torch.float32,
                device=torch.device("cuda"),
            )
            lin_size=lin.shape[0]
            weight=torch.zeros([self.in_channels, self.num_encoding_functions*self.in_channels], dtype=torch.float32, device=torch.device("cuda") )
            for i in range(self.in_channels):
                weight[i:i+1,   i*lin_size:i*lin_size+lin_size ] = lin

            weight=weight.t().contiguous()

            self.conv.weight=torch.nn.Parameter(weight)
            self.weights_initialized=True


    def forward(self, x):

        with torch.no_grad():

            x_proj = self.conv(x)

            if self.only_sin:
                return torch.cat([x, torch.sin(x_proj) ], -1)
            else:
                return torch.cat([x, torch.sin(x_proj), torch.cos(x_proj)], -1)

class PositionalEncodingRandFeatures(torch.nn.Module):
    def __init__(self, in_channels, num_encoding_functions, sigma ):
        super(PositionalEncodingRandFeatures, self).__init__()
        self.in_channels=in_channels
        self.num_encoding_functions=num_encoding_functions
        # self.only_sin=only_sin
        self.sigma=sigma

        out_channels=in_channels*self.num_encoding_functions*2

        self.conv= torch.nn.Linear(in_channels, int(out_channels/2), bias=False).cuda()  #in the case we set the weight ourselves
        self.init_weights()


        #we dont train because that causes it to overfit to the input views and not generalize the specular effects to novel views
        self.conv.weight.requires_grad = False

    def init_weights(self):
        with torch.no_grad():
            num_input = self.in_channels
            self.conv.weight.normal_(0, self.sigma )

            self.weights_initialized=True


    def forward(self, x):

        with torch.no_grad():
            x_proj = self.conv(x)
            return torch.cat([x, torch.sin(6.14*x_proj), torch.cos(6.14*x_proj)], -1)





