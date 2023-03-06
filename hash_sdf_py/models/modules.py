import torch
from torch.autograd import Function
from torch import Tensor
from torch.nn import functional as F
import torchvision

import sys
# from instant_ngp_2  import HashTable
from hash_sdf  import HashSDF
import numpy as np
import time
import math
import random
# import torch_scatter
# from instant_ngp_2_py.lattice.lattice_py import LatticePy
# from instant_ngp_2_py.lattice.lattice_funcs import *
# from instant_ngp_2_py.instant_ngp_2.instant_ngp_2_funcs import *
# from instant_ngp_2_py.utils.utils import cosine_easing_window


# import instant_ngp_2_py.utils.utils as utils
# from instant_ngp_2_py.utils.utils import LinearWN
# from instant_ngp_2_py.utils.utils import LinearELR

# import instant_ngp_2_py.utils.resize_right.resize_right as resize_right
# import instant_ngp_2_py.utils.resize_right.interp_methods as interp_methods

# import cv2



#coarse2fine  which slowly anneals the weights of a vector of size nr_values. t is between 0 and 1
class Coarse2Fine(torch.nn.Module):
    def __init__(self, nr_values):  
        super(Coarse2Fine, self).__init__()
        
        self.nr_values=nr_values 
        self.last_t=0

    def forward(self, t):

        alpha=t*self.nr_values #becasue cosine_easing_window except the alpha to be in range 0, nr_values
        window=cosine_easing_window(self.nr_values, alpha)

        self.last_t=t
        # print("storing last t as ",t)
        if t>1.0:
            print("why is t larger than 1")
            exit(1)

        return window

    def get_last_t(self):
        return self.last_t
    

#does a coarse to fine update on this texture by hooking on the gradient and changing it on the backward pass by blurring it so that neighbouring pixels get the saem gradient
class Coarse2FineTexture(torch.nn.Module):
    def __init__(self, iterations_to_finish):
    # def __init__(self, nr_filters, neighbourhood_size, dilation=1):
        super(Coarse2FineTexture, self).__init__()
        self.iter=0
        self.iterations_to_finish=iterations_to_finish
        self.is_easing_finished=False

        self.register_full_backward_hook(coarse2fine_tex_backward_hook)

    def forward(self, texture):

        return texture

#we define this backward hook on a module because we need to dynamicaly change the coarse2fine weighting as the optimizaiton goes and the backward_hook of a tensor doenst allow to receive other arguments apart from the gradient
def coarse2fine_tex_backward_hook(module, grad_input, grad_output) -> Tensor or None:
    # grad_input is a tuple containing only the grad of the texture and has shape h,w,c
    assert len(grad_input)==1, "grad_input should have only one element which is the gradient with respect to the texture"
    assert grad_input[0].shape[0]==grad_input[0].shape[1], "grad_input should be square because we assume that for the downsample and upsample"


    grad=grad_input[0]
    max_dimension=max(grad.shape[0], grad.shape[1])
    dimension_lowest=2 #we downsample down to 2x2 img
    scale_lowest=dimension_lowest/max_dimension #we scale down to 2x2 pixels

    grad_img=grad.permute(2,0,1).unsqueeze(0)

    #calculate the scale to which we downsample
    # cur_scale=map_range_val(module.iter, 0, module.iterations_to_finish, scale_lowest, 1) #anneal between scale_lowest and scale=1
    cur_dimension_downscale=map_range_val(module.iter, 0, module.iterations_to_finish, dimension_lowest, max_dimension)
    # print("")
    # print("cur_scale", cur_scale)
    # print("cur_dimension_downscale", cur_dimension_downscale)
    # print("max dimension", max_dimension)
    # # sanity=cur_dimension_output/cur_scale
    # # print("san", sanity)
    # scale_for_upscaling=1/cur_scale
    # print("scale_for_upscaling", scale_for_upscaling)
    # sanity_check=cur_dimension_downscale*scale_for_upscaling
    # print("sanity_check", sanity_check, " which should be close to ", max_dimension)


    #to blur the gradient,  we downsample and then upsample it
    if cur_dimension_downscale<max_dimension:
        # grad_down=resize_right.resize(grad_img, scale_factors=cur_scale, antialiasing=True, interp_method=interp_methods.linear, pad_mode='replicate')
        grad_down=resize_right.resize(grad_img, out_shape=[ math.floor(cur_dimension_downscale), math.floor(cur_dimension_downscale)], antialiasing=True, interp_method=interp_methods.linear, pad_mode='replicate')
        grad_img_back_up=resize_right.resize(grad_down, out_shape=grad_img.shape, antialiasing=True, interp_method=interp_methods.linear, pad_mode='replicate')
        new_grad=grad_img_back_up.permute(0,2,3,1).squeeze(0)
    else:
        new_grad=grad
        module.is_easing_finished=True


    new_grad_tuple=(new_grad,) #make a tuple with only one element

    module.iter+=1

    return new_grad_tuple



#does a coarse to fine update on this texture by hooking on the gradient and changing it on the backward pass by blurring it so that neighbouring pixels get the saem gradient
class Coarse2Fine3DGrid(torch.nn.Module):
    def __init__(self, min_downscale, iterations_to_finish):
    # def __init__(self, nr_filters, neighbourhood_size, dilation=1):
        super(Coarse2Fine3DGrid, self).__init__()
        self.min_downscale=min_downscale
        self.iter=0
        self.iterations_to_finish=iterations_to_finish
        self.is_easing_finished=False

        self.register_full_backward_hook(coarse2fine_3dgrid_backward_hook)

    def forward(self, grid):

        return grid

    #https://stackoverflow.com/a/71536369
    def gaussian_blur(self,vol):
        ks=5
        blur_sigma=2
        # # 3D convolution
        # # vol_in = vol.reshape(1, 1, *vol.shape)
        # k = torch.from_numpy(cv2.getGaussianKernel(ks, blur_sigma)).squeeze().float().cuda()
        # # k3d = torch.einsum('i,j,k->ijk', k, k, k)
        # # k3d = k3d / k3d.sum()
        # # vol_3d = F.conv3d(vol_in, k3d.reshape(1, 1, *k3d.shape), stride=1, padding=len(k) // 2)

        # # Separable 1D convolution
        # k1d = k.view(1, 1, -1)
        # for _ in range(3):
        #     vol = F.conv1d(vol.reshape(-1, 1, vol.size(2)), k1d, padding=ks // 2).view(*vol.shape)
        #     vol = vol.permute(2, 0, 1)

        k = torch.from_numpy(cv2.getGaussianKernel(ks, blur_sigma)).squeeze().float().cuda()

        vol_in = vol
        k1d = k[None, None, :, None, None]
        for i in range(3):
            # print("vol",vol_in.shape)
            vol_in = vol_in.permute(0, 1, 4, 2, 3)
            vol_in = F.conv3d(vol_in, k1d, stride=1, padding=(len(k) // 2, 0, 0))
        vol = vol_in

        return vol

    def box_blur(self,vol):
        ks=5
        blur_sigma=2
        # # 3D convolution
        # # vol_in = vol.reshape(1, 1, *vol.shape)
        # k = torch.from_numpy(cv2.getGaussianKernel(ks, blur_sigma)).squeeze().float().cuda()
        # # k3d = torch.einsum('i,j,k->ijk', k, k, k)
        # # k3d = k3d / k3d.sum()
        # # vol_3d = F.conv3d(vol_in, k3d.reshape(1, 1, *k3d.shape), stride=1, padding=len(k) // 2)

        # # Separable 1D convolution
        # k1d = k.view(1, 1, -1)
        # for _ in range(3):
        #     vol = F.conv1d(vol.reshape(-1, 1, vol.size(2)), k1d, padding=ks // 2).view(*vol.shape)
        #     vol = vol.permute(2, 0, 1)

        k = torch.from_numpy(cv2.getGaussianKernel(ks, blur_sigma)).squeeze().float().cuda()
        k=torch.ones_like(k)/ks

        vol_in = vol
        k1d = k[None, None, :, None, None]
        for i in range(3):
            # print("vol",vol_in.shape)
            vol_in = vol_in.permute(0, 1, 4, 2, 3)
            vol_in = F.conv3d(vol_in, k1d, stride=1, padding=(len(k) // 2, 0, 0))
        vol = vol_in

        return vol

    def avg_pool_strided(self,vol):
        vol=torch.nn.functional.avg_pool3d(vol, kernel_size=2, stride=2)
        vol=vol*9

        return vol


#we define this backward hook on a module because we need to dynamicaly change the coarse2fine weighting as the optimizaiton goes and the backward_hook of a tensor doenst allow to receive other arguments apart from the gradient
def coarse2fine_3dgrid_backward_hook(module, grad_input, grad_output) -> Tensor or None:
    # grad_input is a tuple containing only the grad of the texture and has shape h,w,c
    assert len(grad_input)==1, "grad_input should have only one element which is the gradient with respect to the texture"
    assert grad_input[0].shape[0]==grad_input[0].shape[1], "grad_input should be square because we assume that for the downsample and upsample"


    grad=grad_input[0]
    max_dimension=max(grad.shape[2], grad.shape[3], grad.shape[4])
    dimension_lowest=module.min_downscale #we downsample down to 2x2 img
    scale_lowest=dimension_lowest/max_dimension #we scale down to 2x2 pixels

    is_power_of_2=math.log(max_dimension, 2).is_integer()
    if not is_power_of_2:
        print("my lord why are you making me downscale a grid that is not power of 2. Just make it power of 2, I don't want to deal with weird dimensions!")
        exit(1)

    #I also assume allt he dimensions are the same
    if(grad.shape[2]!= grad.shape[3] or  grad.shape[2]!=grad.shape[4]):
        print("the dimensiosn are not the same but for downsampling I assume they are")
        exit(1)

    # print("max_dimension", max_dimension)

    #calculate the scale to which we downsample
    #with this we actually spend more time on the finer scales because going from scale of 128 to 256 takes a lot more iterations than for scale 2 to 4. so we actually want more of an equal nr of iterationf for each scale 
    # cur_dimension_downscale=map_range_val(module.iter, 0, module.iterations_to_finish, dimension_lowest, max_dimension)
    # print("cur_dimension_downscale float ", cur_dimension_downscale)

    #this actually spend the same amount of time in each scale
    geomspace=np.geomspace(dimension_lowest, max_dimension, num= module.iterations_to_finish)
    if module.iter<module.iterations_to_finish:
        scale_geom=geomspace[module.iter]
    else:
        scale_geom=max_dimension
    cur_dimension_downscale=scale_geom


    

    #make it  a power of 2
    def next_power_of_2(x):  
        return 1 if x == 0 else 2**(x - 1).bit_length()

    # Compute a power of two less than or equal to `n`
    def findPreviousPowerOf2(n):
    
        # set all bits after the last set bit
        n = n | (n >> 1)
        n = n | (n >> 2)
        n = n | (n >> 4)
        n = n | (n >> 8)
        n = n | (n >> 16)
    
        # drop all but the last set bit from `n`
        return n - (n >> 1)

    cur_dimension_downscale=next_power_of_2(int(cur_dimension_downscale) )
    # cur_dimension_downscale=findPreviousPowerOf2(int(cur_dimension_downscale) )
    # print("")
    # print("cur_scale", cur_scale)
    # print("cur_dimension_downscale", cur_dimension_downscale)
    # print("module iter is ",module.iter)
    # print("max dimension", max_dimension)
    # # sanity=cur_dimension_output/cur_scale
    # # print("san", sanity)
    # scale_for_upscaling=1/cur_scale
    # print("scale_for_upscaling", scale_for_upscaling)
    # sanity_check=cur_dimension_downscale*scale_for_upscaling
    # print("sanity_check", sanity_check, " which should be close to ", max_dimension)

    # cur_dimension_downscale=8


    #to blur the gradient,  we downsample and then upsample it
    if cur_dimension_downscale<max_dimension:
        # grad_down=resize_right.resize(grad_img, scale_factors=cur_scale, antialiasing=True, interp_method=interp_methods.linear, pad_mode='replicate')

        # grad_down=resize_right.resize(grad, out_shape=[ math.floor(cur_dimension_downscale), math.floor(cur_dimension_downscale)], antialiasing=True, interp_method=interp_methods.linear, pad_mode='replicate')
        # print("grad_down",grad_down.shape)
        # grad_img_back_up=resize_right.resize(grad_down, out_shape=grad_img.shape, antialiasing=True, interp_method=interp_methods.linear, pad_mode='replicate')
        # new_grad=grad_img_back_up.permute(0,2,3,1).squeeze(0)


        # grad_down=torch.nn.functional.interpolate(grad,size=cur_dimension_downscale, mode="area")
        # print("grad_down",grad_down.shape)
        # grad_back_up=torch.nn.functional.interpolate(grad_down,size=max_dimension, mode="trilinear", align_corners=False)

        scale_factor=cur_dimension_downscale/max_dimension
        # print("scale_factor is ", scale_factor)


        # grad_down=resize_right.resize(grad, out_shape=[ math.floor(cur_dimension_downscale), math.floor(cur_dimension_downscale), math.floor(cur_dimension_downscale)], antialiasing=True, by_convs=False, interp_method=interp_methods.linear, pad_mode='constant', denormalize_weights=True)
        grad_down=resize_right.resize_3d_grid(grad, scale_factors=[scale_factor,scale_factor,scale_factor], out_shape=[ math.floor(cur_dimension_downscale), math.floor(cur_dimension_downscale), math.floor(cur_dimension_downscale)], antialiasing=True, by_convs=False, interp_method=interp_methods.linear, pad_mode='constant', denormalize_weights=False)
        
        # grad_down=torch.nn.functional.interpolate(grad,size=cur_dimension_downscale, mode="trilinear", align_corners=False)
        # print("grad_down",grad_down.shape)
        # grad_back_up=resize_right.resize(grad_down, out_shape=grad.shape, antialiasing=True, interp_method=interp_methods.linear, pad_mode='replicate')
        grad_back_up=torch.nn.functional.interpolate(grad_down,size=max_dimension, mode="trilinear", align_corners=False)

        #attempt 2do it by running a bunch of gaussian convolutions
        # for i in range( int(map_range_val(module.iter, 0, module.iterations_to_finish, 60, 0)) ):
        # print("before blur grad min max is ", grad.min(), grad.max())
        # for i in range( 3 ):
        #     ones=(grad.abs()>0.0).float()

        #     # grad=module.gaussian_blur(grad)
        #     grad=module.avg_pool_strided(grad)
        #     print("grad after blut is ", grad.shape)

        #     # ones=module.gaussian_blur(ones)
        #     # ones=module.box_blur(ones)

        #     # valid=ones>0.0
        #     # grad[valid]=grad[valid]/ones[valid]
        # grad=torch.nn.functional.interpolate(grad,size=max_dimension, mode="trilinear", align_corners=False)
        # print("after blur grad min max is ", grad.min(), grad.max())


        # exit(1)


        #try again by downsampling with nearest and upsampling
        # list_of_downsamples=[]
        # lvl_lower=max_dimension/2
        # for i in range(10):
        #     if lvl_lower>4:
        #         list_of_downsamples.append(lvl_lower)
        #     lvl_lower=lvl_lower/2
        # print("list_of_downsamples",list_of_downsamples)
        # list_of_grid_downsamples=[]
        # grad_downsampled=grad
        # for i in range(len(list_of_downsamples)):
        #     lvl=list_of_downsamples[i]
        #     print("lvl",lvl)
        #     grad_downsampled=torch.nn.functional.interpolate(grad_downsampled, size=int(lvl), mode="nearest")
        #     list_of_grid_downsamples.append(grad_downsampled)
        #     print("appended",grad_downsampled.shape)
        # #upsample with nearest and add together
        # grad_added_together=0
        # for i in range(len(list_of_downsamples)):
        #     grad_downsampled=list_of_grid_downsamples[i]
        #     grad_added_together+=torch.nn.functional.interpolate(grad_downsampled, size=max_dimension, mode="trilinear")
        # #set it
        # grad_back_up=grad_added_together







        #scale the gradient
        # grad_scaling=float(cur_dimension_downscale)/max_dimension
        # grad_scaling=grad_scaling*grad_scaling*grad_scaling
        # print("grad_scaling",grad_scaling)
        # grad_back_up=grad_back_up*grad_scaling #this definitelly makes it better

        new_grad=grad_back_up
        # new_grad=grad

        # new_grad=grad
    else:
        new_grad=grad
        module.is_easing_finished=True


    new_grad_tuple=(new_grad,) #make a tuple with only one element

    module.iter+=1

    return new_grad_tuple




        
#retruns a 1D vector of the indices that of the rows we pick from a matrix of NxM. Then we use the RowPicker to do the index_select
class RandRowPicker(torch.nn.Module):
    def __init__(self, with_replacement=True):  
        super(RandRowPicker, self).__init__()

        self.with_replacement=with_replacement #we default to true because it is significantly faster

        self.first_time=True
        self.full_tensor_nr_rows=0
    def forward(self, full_tensor, nr_samples, pick_all):

        assert full_tensor.dim()==2, "Tensor should have 2 dimensions only"

        if self.first_time:
            self.first_time=False
            self.full_tensor_nr_rows=full_tensor.shape[0]
            self.probabilities = torch.ones([self.full_tensor_nr_rows], dtype=torch.float32, device=torch.device("cuda")) #uniform probabilities to pick every row

        assert self.full_tensor_nr_rows==full_tensor.shape[0], "full_tensor rows does not correspond with the precomputed one"


        #pick random indices
        if pick_all:
            return None
        else:
            indices=torch.multinomial(self.probabilities, nr_samples, replacement=self.with_replacement) #size of the chunk size we selected


        return indices

#samples the rows from an tensor of size NxM given some indices, or return the fulltensor if the indices are None
class RowSampler(torch.nn.Module):
    def __init__(self):  
        super(RowSampler, self).__init__()
        
    def forward(self, full_tensor, indices):


        assert full_tensor.dim()==2, "full_tensor should have dim 2"


        if indices==None:
            return full_tensor

        selected_rows=torch.index_select( full_tensor, dim=0, index=indices) 


        return selected_rows



#returns a 1D vector of the pixels that we pick from the image. We assume that the image is flattened into a 1D array. We assume the flattening is done row by row so index 0 is the top left pixel. Can also return None if we choose to sample the full image densely
class RandPixelPicker(torch.nn.Module):
    def __init__(self, low_discrepancy ):  
        super(RandPixelPicker, self).__init__()

        # self.with_replacement=with_replacement #we default to true because it is significantly faster

        # self.first_time=True
        # self.width=None
        # self.height=None
        # self.probabilities_dict={} #key is the nr of pixels, the val is the probabilities for choosing a pixel for a frame of that size
        self.low_discrepancy=low_discrepancy

    def forward(self, frame, nr_samples, pick_all_pixels):

        # if len(self.probabilities_dict)>50:
        #     print("We have a list of probabilities of ", len(self.probabilities_dict), " and this uses quite some memory. If you are sure this not an issue please ignore")

        #if we don't have probabilities for this size of frame, we add it
        nr_rows=frame.width*frame.height
        # if nr_rows not in self.probabilities_dict and not pick_all_pixels:
        #     print("adding RandPixelPicker probabilities for ", nr_rows)
        #     self.probabilities_dict[nr_rows]= torch.ones([nr_rows], dtype=torch.float32, device=torch.device("cuda")) #uniform probabilities to pick every pixel
    



        #pick random indices
        if pick_all_pixels:
            return None
        else:
            # assert self.width==frame.width, "Frame width does not correspond with the precomputed one"
            # assert self.height==frame.height, "Frame height does not correspond with the precomputed one"
            # probabilities=self.probabilities_dict[nr_rows]
            # indices=torch.multinomial(probabilities, nr_samples, replacement=self.with_replacement) #size of the chunk size we selected

            if self.low_discrepancy:
                indices=InstantNGP.low_discrepancy2d_sampling(nr_samples, frame.height, frame.width)
                indices=torch.from_numpy(indices).long().cuda()
            else:
                indices=torch.randint(0,nr_rows, (nr_samples,))


        return indices

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
            # assert self.width==frame.width, "Frame width does not correspond with the precomputed one"
            # assert self.height==frame.height, "Frame height does not correspond with the precomputed one"
            # probabilities=self.probabilities_dict[nr_rows]
            # indices=torch.multinomial(probabilities, nr_samples, replacement=self.with_replacement) #size of the chunk size we selected


            #we create an 1D image of only the indices of each pixels of the image
            nr_pixels=frame.width*frame.height
            full_indices=torch.arange(nr_pixels).view(1,1,frame.height, frame.width)

            

            #we pick a center pixel such that it has enough pixels up and down 
            half_patch=int(patch_size/2)
            # x_idx=torch.randint(half_patch,frame.width-half_patch, (1,))
            # y_idx=torch.randint(half_patch,frame.height-half_patch, (1,))
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
            # assert self.width==frame.width, "Frame width does not correspond with the precomputed one"
            # assert self.height==frame.height, "Frame height does not correspond with the precomputed one"
            # probabilities=self.probabilities_dict[nr_rows]
            # indices=torch.multinomial(probabilities, nr_samples, replacement=self.with_replacement) #size of the chunk size we selected


            #we create an 1D image of only the indices of each pixels of the image
            nr_pixels=frame.width*frame.height
            full_indices=torch.arange(nr_pixels).view(1,1,frame.height, frame.width)

            

            #we pick a center pixel such that it has enough pixels up and down 
            half_patch=int(patch_size/2)
            # x_idx=torch.randint(half_patch,frame.width-half_patch, (1,))
            # y_idx=torch.randint(half_patch,frame.height-half_patch, (1,))
            all_indices_list=[]
            for i in range(nr_patches):

                x_idx=random.randint(0,frame.width-patch_size)
                y_idx=random.randint(0,frame.height-patch_size)

                indices_selected=torchvision.transforms.functional.crop(full_indices, y_idx, x_idx, patch_size, patch_size)

                indices_selected=indices_selected.flatten()
                all_indices_list.append(indices_selected)
            all_indices=torch.cat(all_indices_list,0)

            return all_indices

#concatenation of a patch together with some random rays. the random rays really help with faster convergence
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

        # #start the max_errors and indices_of_max_erros dict
        # if frame.frame_idx not in self.max_errors_dict:
        #     self.max_errors_dict[frame.frame_idx]=torch.zeros([self.nr_error_sampled_pixels])
        #     self.indices_of_max_errors_dict[frame.frame_idx]=torch.zeros([self.nr_error_sampled_pixels]).int()


        # indices_flat=indices.flatten()
        # errors_flat=errors.flatten()

        # errors_sorted, indices_that_sort_errors= torch.sort(errors_flat)
        # indices_sorted=indices_flat[indices_that_sort_errors]

        # #update the max errors and the indices of those max errors
        # max_errors_suparsed=self.max_errors_dict[frame.frame_idx]>errors_sorted
        # self.indices_of_max_errors_dict[frame.frame_idx][max_errors_suparsed]= indices_sorted[max_errors_suparsed]

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

        # print("errors_sorted", errors_sorted)

        #get the first worse offendors nr_error_sampled_pixels
        errors_sorted=errors_sorted[0:self.nr_error_sampled_pixels]
        indices_sorted=indices_sorted[0:self.nr_error_sampled_pixels]

        # print("errors_sorted", errors_sorted)
        # print("self.max_errors_dict[frame.frame_idx]", self.max_errors_dict[frame.frame_idx])
        # print("errors_sorted", errors_sorted.shape)
        # print("indices_sorted", indices_sorted.shape)



        #update the max errors and the indices of those max errors
        max_errors_suparsed=errors_sorted>self.max_errors_dict[frame.frame_idx]
        # print("max_errors_suparsed", max_errors_suparsed)
        self.indices_of_max_errors_dict[frame.frame_idx][max_errors_suparsed]= indices_sorted[max_errors_suparsed]
        self.max_errors_dict[frame.frame_idx][max_errors_suparsed]= errors_sorted[max_errors_suparsed]

        # print("self.indices_of_max_errors_dict[frame.frame_idx][max_errors_suparsed]", self.indices_of_max_errors_dict[frame.frame_idx][max_errors_suparsed])

        # exit()








#samples the pixels from an image of 1, c, h,w given some indices, or return the full image tensor if the indices are None
class PixelSampler(torch.nn.Module):
    def __init__(self):  
        super(PixelSampler, self).__init__()

        
    def forward(self, img, indices):


        assert img.shape[0]==1, "Img should have batch size 1"

        tex=img2tex(img) #hwc
        # print("tex", tex.shape)

        tex_channels=tex.shape[2]
        tex_flattened=tex.view(-1,tex_channels)        

        # print("tex_flattened ", tex_flattened.shape)
        # print("indices max is  ", indices.max())

        if indices==None:
            return tex_flattened

        selected_pixels=torch.index_select( tex_flattened, dim=0, index=indices) 


        return selected_pixels






#creates ray samples given a frame. Precomputes the point in screen space so it assumes that the frames have the same size all the time
#if rand_indices is not none, we select only some pixels in screen space 
class CreateRaySamplesModule(torch.nn.Module):
    def __init__(self, min_depth, max_depth, nr_samples_per_ray):
        super(CreateRaySamplesModule, self).__init__()

        self.min_depth=min_depth
        self.max_depth=max_depth
        self.nr_samples_per_ray=nr_samples_per_ray

        #precomputed things
        self.first_time=True
        self.width=None
        self.height=None
    def forward(self, frame, rand_indices, randomize_depth, depth_multiplier, fixed_depth=None):

        if self.first_time:
            self.first_time=False
            self.width=frame.width
            self.height=frame.height

            #create the 2D points as h,w,3 where for each pixel we store the x,y,1 coordinate
            #shift by 0.5 in order to hit the pixel center
            x_coord= torch.arange(frame.width).view(-1, 1, 1).repeat(1,frame.height, 1)+0.5 #width x height x 1
            y_coord= torch.arange(frame.height).view(1, -1, 1).repeat(frame.width, 1, 1)+0.5 #width x height x 1
            ones=torch.ones(frame.width, frame.height).view(frame.width, frame.height, 1)
            self.points_2D=torch.cat([x_coord, y_coord, ones],2).transpose(0,1).reshape(-1,3).cuda() #Nx3 we tranpose because we want x cooridnate to be inner most so that we traverse row-wise the image

            # print("points 2d", self.points_2D)
            # exit(1)

            #create grid screen space which  has shape W,H,3 where for each pixel we store the x,y,1 coordinate
            # x_coord= torch.arange( width, dtype=torch.float32, device=torch.device("cuda") ).view(width,1,1).repeat(1,height,1)
            # # x_coord=x_coord/frame.width
            # y_coord= torch.arange( height, dtype=torch.float32, device=torch.device("cuda") ).view(1,height,1).repeat(width,1,1)
            # y_coord=height-1-y_coord #flip because opencv
            # # y_coord=y_coord/frame.width
            # ones=torch.ones_like(x_coord)
            # self.grid_screen_space=torch.cat([x_coord,y_coord,ones],2)

            # print("points 2d ", points_2D.shape)

        assert self.width==frame.width, "Frame width does not correspond with the precomputed one"
        assert self.height==frame.height, "Frame height does not correspond with the precomputed one"



        #get 2d points
        selected_points_2D=self.points_2D
        if rand_indices!=None:
            selected_points_2D=torch.index_select( self.points_2D, dim=0, index=rand_indices) 



        #create points in 3D
        K_inv=torch.from_numpy( np.linalg.inv(frame.K) ).to("cuda").float()
        #get from screen to cam coords
        pixels_selected_screen_coords_t=selected_points_2D.transpose(0,1) #3xN
        pixels_selected_cam_coords=torch.matmul(K_inv,pixels_selected_screen_coords_t).transpose(0,1)

        #multiply at various depths
        nr_rays=pixels_selected_cam_coords.shape[0]
        # rand_depth=torch.rand(nr_rays, self.nr_samples_per_ray, 1) *(self.min_depth - self.max_depth) +self.max_depth
        # pixels_selected_cam_coords=min_depth
        # print("rand_depth", rand_depth.min(), rand_depth.max())

        depth_slices=torch.linspace(self.min_depth, self.max_depth, self.nr_samples_per_ray).view(1,-1,1)
        if randomize_depth:
            depth_slice_width = (self.max_depth-self.min_depth) /  (self.nr_samples_per_ray-1) #size of the bucker in which each sample falls
            depth_slice_width=depth_slice_width*0.999 #to ensure that the points from different depth sliced don't touch
            half_movemment=depth_slice_width/2
            rand_depth=torch.rand(nr_rays, self.nr_samples_per_ray, 1) * depth_slice_width - half_movemment
            depth_slices=depth_slices+rand_depth
        else:
            depth_slices=depth_slices.repeat(nr_rays, 1,1)


        # print("depth_slices is ", depth_slices.shape)
        if depth_multiplier!=None:
            # print("depth_multiplier", depth_multiplier.shape)
            # depth_slices=depth_slices*depth_multiplier
            #ged the delta between each slice #the slices are in camera coordinates and they store the depth of each layer of the rays
            depth_slices_diffs=depth_slices 
            depth_slices_diffs[:, 1:, :] = depth_slices_diffs[:, 1:, :] - depth_slices_diffs[:, :-1, :]
            depth_slices_diffs=depth_slices_diffs*depth_multiplier #each layer is shortened or widened
            depth_slices=torch.cumsum(depth_slices_diffs, dim=1)
            # print("depth_slices", depth_slices.shape)

        #if we have a fixed depth value, we set that one
        if fixed_depth is not None:
            assert self.nr_samples_per_ray, "For a fixed depth we assume we have only 2 samples per ray. For implementation reasons. the volumetric integration doesnt work yet for just one sample"
            fixed_depth=fixed_depth.view(-1,1)
            depth_slices=fixed_depth.view(-1,1,1).repeat(1,self.nr_samples_per_ray,1)
            depth_slices[:, 1, :] +=1e-4 #slightly shift the next layer


        pixels_selected_cam_coords_repeted=pixels_selected_cam_coords.view(nr_rays, 1, 3).repeat(1, self.nr_samples_per_ray, 1)
        pixels_selected_cam_coords_repeted=pixels_selected_cam_coords_repeted*depth_slices
        pixels_selected_cam_coords=pixels_selected_cam_coords_repeted.view(-1,3)



        #get from cam_coords to world_coords
        tf_world_cam=frame.tf_cam_world.inverse()
        R=torch.from_numpy( tf_world_cam.linear() ).to("cuda").float()
        t=torch.from_numpy( tf_world_cam.translation() ).to("cuda").view(1,3).float()
        pixels_selected_world_coords=torch.matmul(R, pixels_selected_cam_coords.transpose(0,1).contiguous() ).transpose(0,1).contiguous()  + t
        # print("pixels_selected_world_coords", pixels_selected_world_coords.shape)
        #get direction
        pixels_selected_dirs = pixels_selected_world_coords-t
        pixels_selected_dirs=F.normalize(pixels_selected_dirs, p=2, dim=1)

   


        pixels_selected_world_coords=pixels_selected_world_coords.view(nr_rays, self.nr_samples_per_ray, 3)
        pixels_selected_dirs=pixels_selected_dirs.view(nr_rays, self.nr_samples_per_ray, 3)

        

        return pixels_selected_world_coords, pixels_selected_dirs, depth_slices


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

        #multiply at various depths
        nr_rays=pixels_selected_cam_coords.shape[0]
        

        pixels_selected_cam_coords=pixels_selected_cam_coords.view(nr_rays, 3)



        #get from cam_coords to world_coords
        tf_world_cam=frame.tf_cam_world.inverse()
        R=torch.from_numpy( tf_world_cam.linear() ).to("cuda").float()
        t=torch.from_numpy( tf_world_cam.translation() ).to("cuda").view(1,3).float()
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

            #set the new weights =
            self.conv.weight=torch.nn.Parameter(weight)
            # self.conv.weight.requires_grad=False
            # print("weight is", weight.shape)
            # print("bias is", self.conv.bias.shape)
            # print("weight is", weight)

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

            # if self.only_sin:
                # return torch.cat([x, torch.sin(6.14*x_proj) ], -1)
            # else:
            return torch.cat([x, torch.sin(6.14*x_proj), torch.cos(6.14*x_proj)], -1)


# class BlockSiren(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, bias=True, activ=torch.sin, is_first_layer=False, scale_init=90 ):
#         super(BlockSiren, self).__init__()
#         self.bias=bias
#         self.activ=activ
#         self.is_first_layer=is_first_layer
#         self.scale_init=scale_init




#         # self.conv= torch.nn.Linear(in_channels, out_channels, bias=self.bias ).cuda()
#         self.conv= LinearWN(in_channels, out_channels, bias=self.bias ).cuda()



#         # if self.activ==torch.sin or self.activ==None:
#         with torch.no_grad():
#             is_wnw = is_weight_norm_wrapped(self.conv)
#             if is_wnw:
#                 self.conv.fuse()

#             if self.activ==torch.sin:
#                     num_input = in_channels
#                     # See supplement Sec. 1.5 for discussion of factor 30
#                     if self.is_first_layer:
#                         self.conv.weight.uniform_(-np.sqrt(6 / num_input) , np.sqrt(6 / num_input) )
#                     else:
#                         self.conv.weight.uniform_(-np.sqrt(6 / num_input) , np.sqrt(6 / num_input) )
#             elif self.activ==None:
#                 # exit(1)
#                 # self.conv.weight.normal_(0, 0.1)
#                 leaky_relu_init(self.conv, negative_slope=1.0)

#             if is_wnw:
#                 self.conv.fuse()


#         self.weights_initialized=True #so that apply_weight_init_fn doesnt initialize anything




#     def forward(self, x):

#         x = self.conv(x )


#         if self.activ==torch.sin:
#             if self.is_first_layer:
#                 print("forst layer x min max is ", x.min(), " ", x.max())
#                 x=self.scale_init*x
#             else:
#                 # x=x*1
#                 pass
#             # print("before activ x min max is ", x.min(), " ", x.max())
#             print("before activ x mean and std is ", x.mean(), " ", x.var())
#             x=self.activ(x)



#         elif self.activ is not None:
#             x=self.activ(x)


#         return x


class BlockSiren2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, activ=torch.sin, is_first_layer=False, scale_init=30 ):
        super(BlockSiren2, self).__init__()
        self.bias=bias
        self.activ=activ
        self.is_first_layer=is_first_layer
        self.scale_init=scale_init




        # self.conv= torch.nn.Linear(in_channels, out_channels, bias=self.bias ).cuda()
        self.conv= LinearWN(in_channels, out_channels, bias=self.bias ).cuda()



        # if self.activ==torch.sin or self.activ==None:
        with torch.no_grad():
            is_wnw = is_weight_norm_wrapped(self.conv)
            if is_wnw:
                self.conv.fuse()

            if self.activ==torch.sin:
                    num_input = in_channels
                    # See supplement Sec. 1.5 for discussion of factor 30
                    if self.is_first_layer:
                        self.conv.weight.uniform_(-1 / num_input, 1 / num_input)
                    else:
                        self.conv.weight.uniform_(-np.sqrt(6 / num_input)/self.scale_init , np.sqrt(6 / num_input)/self.scale_init )
            elif self.activ==None:
                # exit(1)
                # self.conv.weight.normal_(0, 0.1)
                leaky_relu_init(self.conv, negative_slope=1.0)

            if is_wnw:
                self.conv.fuse()


        self.weights_initialized=True #so that apply_weight_init_fn doesnt initialize anything




    def forward(self, x):

        x = self.conv(x )


        if self.activ==torch.sin:
            x=self.activ(self.scale_init*x)



        elif self.activ is not None:
            x=self.activ(x)


        return x



class GaussActiv(torch.nn.Module):
    def __init__(self, sigma ):
        super(GaussActiv, self).__init__()
        self.sigma=sigma

    def forward(self, x):

        x=gauss_activ(x,self.sigma)

        return x



#does a coarse to fine update on this texture by hooking on the gradient and changing it on the backward pass by blurring it so that neighbouring pixels get the saem gradient
class Coarse2FineTexture(torch.nn.Module):
    def __init__(self, iterations_to_finish):
    # def __init__(self, nr_filters, neighbourhood_size, dilation=1):
        super(Coarse2FineTexture, self).__init__()
        self.iter=0
        self.iterations_to_finish=iterations_to_finish
        self.is_easing_finished=False

        self.register_full_backward_hook(coarse2fine_tex_backward_hook)

    def forward(self, texture):

        return texture

#we define this backward hook on a module because we need to dynamicaly change the coarse2fine weighting as the optimizaiton goes and the backward_hook of a tensor doenst allow to receive other arguments apart from the gradient
def coarse2fine_tex_backward_hook(module, grad_input, grad_output) -> Tensor or None:
    # grad_input is a tuple containing only the grad of the texture and has shape h,w,c
    assert len(grad_input)==1, "grad_input should have only one element which is the gradient with respect to the texture"
    assert grad_input[0].shape[0]==grad_input[0].shape[1], "grad_input should be square because we assume that for the downsample and upsample"


    grad=grad_input[0]
    max_dimension=max(grad.shape[0], grad.shape[1])
    dimension_lowest=2 #we downsample down to 2x2 img
    scale_lowest=dimension_lowest/max_dimension #we scale down to 2x2 pixels

    grad_img=grad.permute(2,0,1).unsqueeze(0)

    #calculate the scale to which we downsample
    cur_dimension_downscale=map_range_val(module.iter, 0, module.iterations_to_finish, dimension_lowest, max_dimension)


    #to blur the gradient,  we downsample and then upsample it
    if cur_dimension_downscale<max_dimension:
        # grad_down=resize_right.resize(grad_img, scale_factors=cur_scale, antialiasing=True, interp_method=interp_methods.linear, pad_mode='replicate')
        grad_down=resize_right.resize(grad_img, out_shape=[ math.floor(cur_dimension_downscale), math.floor(cur_dimension_downscale)], antialiasing=True, interp_method=interp_methods.linear, pad_mode='replicate')
        grad_img_back_up=resize_right.resize(grad_down, out_shape=grad_img.shape, antialiasing=True, interp_method=interp_methods.linear, pad_mode='replicate')
        new_grad=grad_img_back_up.permute(0,2,3,1).squeeze(0)
    else:
        new_grad=grad
        module.is_easing_finished=True


    new_grad_tuple=(new_grad,) #make a tuple with only one element

    module.iter+=1

    return new_grad_tuple


class Coarse2FineImg(torch.nn.Module):
    def __init__(self, iterations_to_finish):
    # def __init__(self, nr_filters, neighbourhood_size, dilation=1):
        super(Coarse2FineImg, self).__init__()
        self.iter=0
        self.iterations_to_finish=iterations_to_finish
        self.is_easing_finished=False

        self.register_full_backward_hook(coarse2fine_img_backward_hook)

    def forward(self, texture):

        return texture

#we define this backward hook on a module because we need to dynamicaly change the coarse2fine weighting as the optimizaiton goes and the backward_hook of a tensor doenst allow to receive other arguments apart from the gradient
def coarse2fine_img_backward_hook(module, grad_input, grad_output) -> Tensor or None:
    # grad_input is a tuple containing only the grad of the texture and has shape h,w,c
    assert len(grad_input)==1, "grad_input should have only one element which is the gradient with respect to the texture"
    assert grad_input[0].shape[2]==grad_input[0].shape[3], "grad_input should be square because we assume that for the downsample and upsample"


    grad=grad_input[0]
    max_dimension=max(grad.shape[2], grad.shape[3])
    dimension_lowest=2 #we downsample down to 2x2 img
    scale_lowest=dimension_lowest/max_dimension #we scale down to 2x2 pixels

    grad_img=grad

    #calculate the scale to which we downsample
    cur_dimension_downscale=map_range_val(module.iter, 0, module.iterations_to_finish, dimension_lowest, max_dimension)


    #to blur the gradient,  we downsample and then upsample it
    if cur_dimension_downscale<max_dimension:
        # grad_down=resize_right.resize(grad_img, scale_factors=cur_scale, antialiasing=True, interp_method=interp_methods.linear, pad_mode='replicate')
        grad_down=resize_right.resize(grad_img, out_shape=[ math.floor(cur_dimension_downscale), math.floor(cur_dimension_downscale)], antialiasing=True, interp_method=interp_methods.linear, pad_mode='replicate')
        grad_img_back_up=resize_right.resize(grad_down, out_shape=grad_img.shape, antialiasing=True, interp_method=interp_methods.linear, pad_mode='replicate')
        new_grad=grad_img_back_up
    else:
        new_grad=grad
        module.is_easing_finished=True


    new_grad_tuple=(new_grad,) #make a tuple with only one element

    module.iter+=1

    return new_grad_tuple



