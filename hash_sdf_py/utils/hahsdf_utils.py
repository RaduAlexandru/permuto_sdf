import torch
from typing import Optional
import math
import numpy as np
from collections import namedtuple
import torch.nn.functional as F



# from instant_ngp_2_py.utils.utils import *
from hash_sdf_py.utils.nerf_utils import get_midpoint_of_sections
from hash_sdf_py.utils.sdf_utils import sdf_loss_spheres
from hash_sdf_py.utils.sdf_utils import sdf_loss_sphere
# from instant_ngp_2_py.utils.sphere import SpherePy
from hash_sdf  import OccupancyGrid
from hash_sdf  import VolumeRendering
from hash_sdf  import RaySamplesPacked
from hash_sdf  import RaySampler
from easypbr import Camera #for get_frames_cropped

# from instant_ngp_2_py.schedulers.warmup import *

# import instant_ngp_2_py.losses.robust_loss_pytorch as robust_loss_pytorch

# import apex

def module_exists(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True



def run_sdf_and_color_pred(model, model_rgb, model_bg, lattice, lattice_bg, iter_nr, use_only_dense_grid, cos_anneal_ratio, forced_variance, frame, aabb, occupancy_grid, min_dist_between_samples, max_nr_samples_per_ray, nr_samples_bg, return_curvature, chunk_size, without_mask):
    
    # chunk_size=1000

    ray_origins_full, ray_dirs_full=model.create_rays(frame, rand_indices=None) # ray origins and dirs as nr_pixels x 3
    nr_chunks=math.ceil( ray_origins_full.shape[0]/chunk_size)
    ray_origins_list=torch.chunk(ray_origins_full, nr_chunks)
    ray_dirs_list=torch.chunk(ray_dirs_full, nr_chunks)
    pred_rgb_list=[]
    pred_rgb_fused_list=[]
    pred_rgb_bg_list=[]
    # pred_rgb_view_dep_list=[]
    pred_depth_list=[]
    pred_inv_s_list=[]
    pred_weights_sum_list=[]
    pred_weights_fused_sum_list=[]
    pred_curvature_list=[]
    pred_imp_points_list=[]
    pred_imp_points_neus_list=[]
    pred_normals_vol_fused_list=[]
    for i in range(len(ray_origins_list)):
        ray_origins=ray_origins_list[i]
        ray_dirs=ray_dirs_list[i]
        ray_points_entry, ray_t_entry, ray_points_exit, ray_t_exit, ray_box_intersection=aabb.ray_intersection(ray_origins, ray_dirs)
        nr_rays_vis=ray_origins.shape[0]

        ray_samples_packed=occupancy_grid.compute_samples_in_occupied_regions(ray_origins, ray_dirs, ray_t_entry, ray_t_exit, min_dist_between_samples, max_nr_samples_per_ray, model.training)
        ray_samples_packed=ray_samples_packed.get_valid_samples()


        ####IMPORTANCE sampling
        if ray_samples_packed.samples_pos.shape[0]!=0: #if we actualyl have samples for this batch fo rays
            inv_s_imp_sampling=512
            inv_s_multiplier=1
            do_imp_sampling=False
            if do_imp_sampling:
                sdf_sampled_packed, _, _=model(ray_samples_packed.samples_pos, lattice, iter_nr, use_only_dense_grid=False)
                ray_samples_packed.set_sdf(sdf_sampled_packed) ##set sdf
                alpha=VolumeRendering.sdf2alpha(ray_samples_packed, sdf_sampled_packed, inv_s_imp_sampling, True, inv_s_multiplier)
                transmittance, bg_transmittance= VolumeRendering.cumprod_alpha2transmittance(ray_samples_packed, 1-alpha)
                weights = alpha * transmittance
                weights_sum, weight_sum_per_sample=VolumeRendering.sum_over_each_ray(ray_samples_packed, weights)
                weight_sum_per_sample[weight_sum_per_sample==0]=1e-6 #prevent nans
                weights/=weight_sum_per_sample #prevent nans
                cdf=VolumeRendering.compute_cdf(ray_samples_packed, weights)
                ray_samples_packed_imp=VolumeRendering.importance_sample(ray_origins, ray_dirs, ray_samples_packed, cdf, 16, model.training)
                sdf_sampled_packed_imp, _, _=model(ray_samples_packed_imp.samples_pos, lattice, iter_nr, use_only_dense_grid=False)
                ray_samples_packed_imp.set_sdf(sdf_sampled_packed_imp) ##set sdf
                ray_samples_combined=VolumeRendering.combine_uniform_samples_with_imp(ray_origins, ray_dirs, ray_samples_packed, ray_samples_packed_imp)
                ray_samples_packed=ray_samples_combined#swap
                ray_samples_packed=ray_samples_packed.get_valid_samples() #still need to get the valid ones because we have less samples than allocated
                ####SECOND ITER
                inv_s_multiplier=2
                sdf_sampled_packed=ray_samples_packed.samples_sdf #we already combined them and have the sdf
                alpha=VolumeRendering.sdf2alpha(ray_samples_packed, sdf_sampled_packed, inv_s_imp_sampling, True, inv_s_multiplier)
                transmittance, bg_transmittance= VolumeRendering.cumprod_alpha2transmittance(ray_samples_packed, 1-alpha)
                weights = alpha * transmittance
                weights_sum, weight_sum_per_sample=VolumeRendering.sum_over_each_ray(ray_samples_packed, weights)
                weight_sum_per_sample[weight_sum_per_sample==0]=1e-6 #prevent nans
                weights/=weight_sum_per_sample #prevent nans
                cdf=VolumeRendering.compute_cdf(ray_samples_packed, weights)
                ray_samples_packed_imp=VolumeRendering.importance_sample(ray_origins, ray_dirs, ray_samples_packed, cdf, 16, model.training)
                ray_samples_combined=VolumeRendering.combine_uniform_samples_with_imp(ray_origins, ray_dirs, ray_samples_packed, ray_samples_packed_imp)
                ray_samples_packed=ray_samples_combined#swap
                ray_samples_packed=ray_samples_packed.get_valid_samples() #still need to get the valid ones because we have less samples than allocated
                #####FINISH imp sampling
                pred_imp_points_list.append(ray_samples_packed_imp.samples_pos)


        
        z_vals, last_z_vals = model.ray_sampler.get_z_vals(ray_origins, ray_dirs, model, lattice, iter_nr, use_only_dense_grid) #nr_rays x nr_samples


        #DEBUG get also the samples from neus
        ray_samples_imp_neus = ray_origins[:, None, :] + ray_dirs[:, None, :] * last_z_vals[..., :, None]  # n_rays, n_samples, 3
        pred_imp_points_neus_list.append(ray_samples_imp_neus.view(-1,3))

        z_vals_rgb = get_midpoint_of_sections(z_vals)
        # Section midpoints
        ray_samples_sdf = ray_origins[:, None, :] + ray_dirs[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        ray_samples_rgb = ray_origins[:, None, :] + ray_dirs[:, None, :] * z_vals_rgb[..., :, None]  # n_rays, n_samples, 3
        dirs = ray_dirs[:, None, :].expand(ray_samples_rgb.shape )
        nr_rays=ray_samples_rgb.shape[0]
        nr_samples=ray_samples_rgb.shape[1]

        #new stuff based on neus
        # pts_sdf = ray_samples_sdf.reshape(-1, 3)
        pts = ray_samples_rgb.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)


        if without_mask:
            z_vals_bg, dummy, ray_samples_bg_4d, ray_samples_bg = model_bg.ray_sampler_bg.get_z_vals_bg(ray_origins, ray_dirs, model_bg, lattice_bg, iter_nr) 
            dirs_bg = ray_dirs[:, None, :].expand(ray_samples_bg.shape ).contiguous()

            ray_samples_packed_bg= RaySampler.compute_samples_bg(ray_origins, ray_dirs, ray_t_exit, nr_samples_bg, aabb.m_radius, aabb.m_center_tensor, model.training)


        # #predict sdf
        sdf, sdf_gradients, feat, variance=model.get_sdf_and_gradient(pts, lattice, iter_nr, use_only_dense_grid)
        sdf=sdf.detach()
        sdf_gradients=sdf_gradients.detach()
        feat=feat.detach()

        # #predict rgb
        rgb_samples, rgb_samples_view_dep = model_rgb(model, feat, sdf_gradients, pts, dirs, lattice, iter_nr)
        rgb_samples=rgb_samples.view(nr_rays, -1, 3)
        if rgb_samples_view_dep is not None:
            rgb_samples_view_dep=rgb_samples_view_dep.view(nr_rays, -1, 3)

        #volume render
        weights, inv_s, inv_s_before_exp, bg_transmittance = model.volume_renderer(pts, lattice, z_vals, ray_t_exit, sdf, sdf_gradients, dirs, nr_rays, nr_samples, cos_anneal_ratio, forced_variance=forced_variance) #neus
        # weights, inv_s, inv_s_before_exp, bg_transmittance = model.volume_renderer.forward2(sdf, nr_rays, nr_samples, forced_variance=forced_variance) #neus

        #curvature
        if return_curvature:
            # sdf2, sdf_curvature, feat2, sdf_residual=model.get_sdf_and_curvature_1d_precomputed_gradient( pts, sdf, sdf_gradients, lattice, iter_nr, use_only_dense_grid)
            sdf2, sdf_curvature, feat2=model.get_sdf_and_curvature_1d_precomputed_gradient_normal_based( pts, sdf_gradients, lattice, iter_nr)
            sdf2=sdf2.detach()
            sdf_curvature=sdf_curvature.detach()
            if feat2 is not None:
                feat2=feat2.detach()
            sdf_curvature=sdf_curvature.view(nr_rays, -1, 1)
            # print("sdf_curvature", sdf_curvature.shape)
            # print("rgb_samples", rgb_samples.shape)
            sdf_curvature=sdf_curvature
            pred_curvature= torch.sum(weights.unsqueeze(-1) * sdf_curvature.abs(), 1)
            # print("pred_curvature", pred_curvature.shape)
            pred_curvature_list.append(pred_curvature.detach())



        pred_rgb = torch.sum(weights.unsqueeze(-1) * rgb_samples, 1)




        #####make the samples pack to be the same as the neus samples
        use_samples_from_neus=False
        if use_samples_from_neus:
            ray_samples_packed=RaySamplesPacked(nr_rays, nr_rays*nr_samples)
            ray_samples_packed.rays_have_equal_nr_of_samples=True
            ray_samples_packed.fixed_nr_of_samples_per_ray=nr_samples
            ray_samples_packed.samples_pos=pts.view(-1,3)
            ray_samples_packed.samples_dirs=dirs.view(-1,3)
            ray_samples_packed.samples_z=z_vals_rgb.view(-1,1)
            dists = z_vals_rgb[:, 1:] - z_vals_rgb[:, :-1]
            dists = torch.cat([dists, ray_t_exit - z_vals_rgb[:, -1:]], -1) # included also the dist from the sphere intersection
            ray_samples_packed.samples_dt=dists.view(-1,1)


        ########FUSED ones
        ###FUSED sdf--------------------
        # print("ray_samples_packed.samples_pos",ray_samples_packed.samples_pos)
        if ray_samples_packed.samples_pos.shape[0]==0: #we ended up in a region where no samples where created
            pred_rgb_fused=torch.zeros_like(ray_origins)
            pred_normals_vol_fused=torch.zeros_like(ray_origins)
            weights_sum_fused=torch.zeros_like(ray_origins)[:,0:1]
            if without_mask:
                pred_rgb_bg_list.append( torch.zeros_like(ray_origins) )
        else:
            # pred_rgb_fused=torch.zeros_like(ray_origins)
            sdf_fused, sdf_gradients_fused, feat_fused, _ =model.get_sdf_and_gradient(ray_samples_packed.samples_pos, lattice, iter_nr, use_only_dense_grid=False)
            sdf_fused=sdf_fused.detach()
            sdf_gradients_fused=sdf_gradients_fused.detach()
            feat_fused=feat_fused.detach()

            #FUSED vol render----------------------
            weights_fused, weights_sum_fused, inv_s_fused, inv_s_before_exp_fused, bg_transmittance_fused = model.volume_renderer.vol_render_samples_packed(ray_samples_packed, ray_t_exit, True, sdf_fused, sdf_gradients_fused, cos_anneal_ratio, forced_variance=forced_variance) #neus

            #Fused RGB---------------------------
            rgb_samples_fused, _ = model_rgb(model, feat_fused, sdf_gradients_fused, ray_samples_packed.samples_pos, ray_samples_packed.samples_dirs, lattice, iter_nr, None, None)

            #FUSED integrate weigths and rgb_samples_fused
            # pred_rgb_fused=VolumeRendering.integrate_rgb_and_weights(ray_samples_packed, rgb_samples_fused, weights_fused)
            pred_rgb_fused=model.volume_renderer.integrator_module(ray_samples_packed, rgb_samples_fused, weights_fused)

            #volumetrically integrate also the normals
            pred_normals_vol_fused=model.volume_renderer.integrator_module(ray_samples_packed, sdf_gradients_fused, weights_fused)


            #run nerf bg
            if without_mask:
                #the way neus does it
                rgb_samples_bg, density_samples_bg=model_bg( ray_samples_bg_4d.view(-1,4), dirs_bg.view(-1,3), lattice_bg, iter_nr) 
                rgb_samples_bg=rgb_samples_bg.view(nr_rays, nr_samples_bg,3)
                density_samples_bg=density_samples_bg.view(nr_rays, nr_samples_bg)
                # #get weights for the integration
                weights_bg, disp_map_bg, acc_map_bg, depth_map_bg, _=model_bg.volume_renderer(density_samples_bg, z_vals_bg, None)
                pred_rgb_bg = torch.sum(weights_bg.unsqueeze(-1) * rgb_samples_bg, 1)
                #combine
                pred_rgb_bg = bg_transmittance.view(-1,1) * pred_rgb_bg
                pred_rgb = pred_rgb + pred_rgb_bg


                # rgb_samples_bg, density_samples_bg=model_bg( ray_samples_packed_bg.samples_pos_4d.view(-1,4), ray_samples_packed_bg.samples_dirs.view(-1,3), lattice_bg, iter_nr) 
                # rgb_samples_bg=rgb_samples_bg.view(nr_rays, nr_samples_bg,3)
                # density_samples_bg=density_samples_bg.view(nr_rays, nr_samples_bg)
                # # #get weights for the integration
                # # weights_bg, disp_map_bg, acc_map_bg, depth_map_bg, _=model_bg.volume_renderer(density_samples_bg, z_vals_bg, None)
                # pred_rgb_bg, pred_depth_bg, _= model_bg.volume_renderer_general.volume_render_nerf(ray_samples_packed_bg, rgb_samples_bg.view(-1,3), density_samples_bg.view(-1,1), ray_t_exit, False)
                # #combine attempt 3 like in https://github.com/lioryariv/volsdf/blob/a974c883eb70af666d8b4374e771d76930c806f3/code/model/network_bg.py#L96
                # pred_rgb_bg = bg_transmittance_fused * pred_rgb_bg
                # pred_rgb_fused = pred_rgb_fused + pred_rgb_bg

                pred_rgb_bg_list.append(pred_rgb_bg.detach())

                #debug 
                # weights_sum=weights.sum(dim=1, keepdim=True)
                # bg_transmittance_inv=1.0-bg_transmittance
                # diff=(weights_sum.flatten() - bg_transmittance_inv.flatten())**2
                # print("diff is ",diff.mean())


    


        # pred_rgb_view_dep = torch.sum(weights.unsqueeze(-1) * rgb_samples_view_dep, 1)
        pred_depth= torch.sum(weights.unsqueeze(-1) * z_vals_rgb.unsqueeze(-1), 1)

        pred_rgb_list.append(pred_rgb.detach())
        pred_rgb_fused_list.append(pred_rgb_fused.detach())
        pred_normals_vol_fused_list.append(pred_normals_vol_fused.detach())
        # pred_rgb_view_dep_list.append(pred_rgb_view_dep.detach())
        pred_depth_list.append(pred_depth.detach())
        pred_weights_sum_list.append(  torch.sum(weights.unsqueeze(-1), 1)  )
        pred_weights_fused_sum_list.append(  weights_sum_fused.view(-1,1)  )

    pred_rgb=torch.cat(pred_rgb_list,0)
    pred_rgb_fused=torch.cat(pred_rgb_fused_list,0)
    pred_normals_vol_fused=torch.cat(pred_normals_vol_fused_list,0)
    # pred_rgb_view_dep=torch.cat(pred_rgb_view_dep_list,0)
    pred_depth=torch.cat(pred_depth_list,0)
    pred_weights_sum=torch.cat(pred_weights_sum_list,0)
    pred_weights_fused_sum=torch.cat(pred_weights_fused_sum_list,0)
    if return_curvature:
        pred_curvature=torch.cat(pred_curvature_list,0)
    #vis
    pred_rgb_img=lin2nchw(pred_rgb, frame.height, frame.width)
    pred_rgb_fused_img=lin2nchw(pred_rgb_fused, frame.height, frame.width)
    pred_normals_vol_fused_img=lin2nchw(pred_normals_vol_fused, frame.height, frame.width)
    # pred_rgb_view_dep_img=lin2nchw(pred_rgb_view_dep, frame.height, frame.width)
    pred_depth_img=lin2nchw(pred_depth, frame.height, frame.width)
    pred_weights_sum_img=lin2nchw(pred_weights_sum, frame.height, frame.width)
    pred_weights_fused_sum_img=lin2nchw(pred_weights_fused_sum, frame.height, frame.width)
    if return_curvature:
        pred_curvature_img=lin2nchw(pred_curvature, frame.height, frame.width)
    else:
        pred_curvature_img=None

    if without_mask:
        pred_rgb_bg=torch.cat(pred_rgb_bg_list,0)
        pred_rgb_bg_img=lin2nchw(pred_rgb_bg, frame.height, frame.width)
    else:
        pred_rgb_bg_img=None

    if len(pred_imp_points_list)!=0:
        pred_imp_points=torch.cat(pred_imp_points_list,0)
        show_points(pred_imp_points,"pred_imp_points")
    pred_imp_points_neus=torch.cat(pred_imp_points_neus_list,0)
    show_points(pred_imp_points_neus,"pred_imp_points_neus")

    # print("pred_curvature_img", pred_curvature_img.shape)
    # Gui.show(tensor2mat(pred_rgb_img).rgb2bgr(), "pred_rgb_img_control")
    # Gui.show(tensor2mat(pred_depth_img), "pred_depth_img_control")
    # Gui.show(tensor2mat(pred_weights_sum_img), "pred_weights_sum_img")
    torch.cuda.empty_cache() 
    #show also the ray end after volumetric integration 
    ray_end_volumetric = ray_origins_full + ray_dirs_full * pred_depth.view(-1,1)
    # show_points(ray_end_volumetric, "ray_end_volumetric")


    Gui.show(tensor2mat(pred_weights_fused_sum_img), "pred_weights_fused_sum_img")


    # return pred_rgb_img, pred_rgb_view_dep_img, pred_depth_img, pred_weights_sum_img, ray_end_volumetric, pred_curvature_img
    return pred_rgb_img, pred_rgb_fused_img, pred_depth_img, pred_weights_sum_img, ray_end_volumetric, pred_curvature_img, pred_rgb_bg_img, pred_normals_vol_fused_img


def init_losses():
    loss=0
    loss_rgb=torch.tensor(0)
    loss_eikonal=torch.tensor(0).float()
    loss_curvature=torch.tensor(0)
    loss_lipshitz=torch.tensor(0)

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
        print("dataset not known")
        exit()

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


def train(args, config_path, loader_train, frames_train, experiment_name, dataset_name, with_viewer, with_tensorboard, save_checkpoint, checkpoint_path, tensor_reel, lr, iter_start_curv, lr_milestones, iter_finish_training):

    #things needed for trainin
    from instant_ngp_2  import NGPGui
    from instant_ngp_2  import Lattice
    from instant_ngp_2  import InstantNGP
    from instant_ngp_2_py.callbacks.callback_utils import create_callbacks
    from instant_ngp_2_py.callbacks.callback_utils import create_callbacks_simpler_1
    from instant_ngp_2_py.callbacks.phase import Phase
    from instant_ngp_2_py.instant_ngp_2.models import SDF
    from instant_ngp_2_py.instant_ngp_2.models import SDFDenseAndHash
    from instant_ngp_2_py.instant_ngp_2.models import RGB
    from instant_ngp_2_py.instant_ngp_2.models import NerfHash
    from instant_ngp_2_py.instant_ngp_2.models import Colorcal
    from instant_ngp_2_py.optimizers.radam import RAdam
    from instant_ngp_2_py.schedulers.multisteplr import MultiStepLR
    from instant_ngp_2_py.optimizers.grad_scaler import GradScaler
    from instant_ngp_2_py.instant_ngp_2.sdf_utils import sdf_loss_sphere
    from instant_ngp_2_py.instant_ngp_2.sdf_utils import sdf_loss_spheres
    from instant_ngp_2_py.instant_ngp_2.sdf_utils import sphere_trace
    from instant_ngp_2_py.instant_ngp_2.sdf_utils import filter_unconverged_points
    from instant_ngp_2_py.instant_ngp_2.sdf_utils import sample_sdf_in_layer

    if with_viewer:
        view=Viewer.create(config_path)
        ngp_gui=NGPGui.create(view)
    

    first_time=True

    if first_time and with_viewer:
        view.m_camera.from_string("-1.69051 0.499783 0.824015  -0.119454    -0.5118 -0.0721315 0.847683  0.0532509 -0.0668205 -0.0933657 60 0.0502494 5024.94")
        # zoomint to eh base of the bear to see the error
        # view.m_camera.from_string(" -0.380117 -0.0556938   0.167511  -0.105197  -0.426046 -0.0499672 0.897171  0.116958 -0.208198 -0.237689 60 0.0502494 5024.94")

    
    #params
    nr_lattice_features=2
    nr_resolutions=24
    # nr_resolutions=8
    use_home= args.type=="home"
    use_mask_loss=not args.without_mask
    nr_samples_bg=32
    print("use_all_imgs", args.use_all_imgs)
    print("without_mask", args.without_mask)


    #torch stuff 
    lattice=Lattice.create(config_path, "lattice")
    nr_lattice_vertices=lattice.capacity()
    #make a lattice for 4D nerf for hte background
    lattice_bg=Lattice.create(config_path, "lattice")
    lattice_bg.set_sigmas_from_string("1.0 4") #the sigma doenst really matter but the 4 matters here
    #loader
    # loader_train, loader_test= create_dataloader(dataset_name, args.scene, config_path, use_home, args.use_all_imgs, args.without_mask)


    aabb, aabb_big = create_bb_for_dataset(dataset_name)
    if with_viewer:
        bb_mesh = create_bb_mesh(aabb) 
        Scene.show(bb_mesh,"bb_mesh")


    nr_rays_to_create = define_nr_rays(loader_train, use_home) 
    expected_samples_per_ray=64+16+16 #we have 64 uniform samples and we run two importance samples
    target_nr_of_samples=512*expected_samples_per_ray


    # #crop frames around the plant
    # if isinstance(loader_train, DataLoaderPhenorobCP1):
    #     frames_train=get_frames_cropped(loader_train, aabb)
        


            


    # if isinstance(loader_train, DataLoaderPhenorobCP1):
    #     tensor_reel=MiscDataFuncs.frames2tensors(frames_train) #make an tensorreel and get rays from all the images at
    # else:
    #     tensor_reel=MiscDataFuncs.frames2tensors(loader_train.get_all_frames()) #make an tensorreel and get rays from all the images at


    cb=create_callbacks_simpler_1(with_viewer, with_tensorboard, experiment_name, config_path)


    #create phases
    phases= [
        Phase('train', loader_train, grad=True),
        # Phase('test', loader_test, grad=False),
    ]
    # if isinstance(loader_train, DataLoaderPhenorobCP1):
        # phases[0].frames=frames_train 
        # phases[1].frames=frames_train 
    #model 
    model=SDF(nr_lattice_vertices, nr_lattice_features, nr_resolutions=nr_resolutions, boundary_primitive=aabb_big, feat_size_out=32, nr_iters_for_c2f=10000, N_initial_samples=64, N_samples_importance=64, N_iters_upsample=4 ).to("cuda")
    # model=SDFDenseAndHash(nr_lattice_vertices, nr_lattice_features, nr_resolutions=nr_resolutions, boundary_primitive=aabb, feat_size_out=32, nr_iters_for_c2f=1000, N_initial_samples=64, N_samples_importance=64, N_iters_upsample=4).to("cuda")
    model_rgb=RGB(nr_lattice_vertices, nr_lattice_features, nr_resolutions=24, feat_size_in=32, nr_cams=loader_train.nr_samples() ).to("cuda")
    # model_bg=NerfHash(4, nr_lattice_vertices, nr_lattice_features, nr_resolutions=nr_resolutions, boundary_primitive=aabb, nr_samples_per_ray=nr_samples_bg, nr_iters_for_c2f=10000 ).to("cuda")
    model_bg=NerfHash(4, nr_lattice_vertices, nr_lattice_features, nr_resolutions=nr_resolutions, boundary_primitive=aabb, nr_samples_per_ray=nr_samples_bg, nr_iters_for_c2f=1 ).to("cuda")
    model_colorcal=Colorcal(loader_train.nr_samples(), 0)
    occupancy_grid=OccupancyGrid(256, 1.0, [0,0,0])
    # occupancy_grid=OccupancyGrid(128, 1.0, [0,0,0])

    # this tells wandb to log the gradients every log_freq step
    # if(train_params.with_wandb()):
    #     wandb.watch(model, idx = 11, log_freq=100)
    #     wandb.watch(model_rgb, idx=99, log_freq=100)




    first_time_getting_control=True
    run_test=True

    # iter_start_curv=100000
    # iter_start_curv=50000
    # iter_finish_curv=iter_start_curv+10001
    iter_finish_curv=iter_start_curv+1001
    forced_variance_finish_iter=35000
    min_dist_between_samples=0.001
    # min_dist_between_samples=0.00001
    max_nr_samples_per_ray=64


    is_in_training_loop=True


    #load from checkpoint if necessary
    # model.load_state_dict(torch.load("/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/checkpoints/s_379_Default_DEcayCurv50k_RGBWithSlicedFeat_DecayRGBmlpAndLatWD0.1_dtu_scan37/50000/models/sdf_model.pt") )
    # model_rgb.load_state_dict(torch.load("/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/checkpoints/s_379_Default_DEcayCurv50k_RGBWithSlicedFeat_DecayRGBmlpAndLatWD0.1_dtu_scan37/50000/models/rgb_model.pt") )
    # model_bg.load_state_dict(torch.load("/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/checkpoints/s_379_Default_DEcayCurv50k_RGBWithSlicedFeat_DecayRGBmlpAndLatWD0.1_dtu_scan37/50000/models/nerf_hash_bg_model.pt") )
    # model_colorcal.load_state_dict(torch.load("/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/checkpoints/s_379_Default_DEcayCurv50k_RGBWithSlicedFeat_DecayRGBmlpAndLatWD0.1_dtu_scan37/50000/models/colorcal_model.pt") )
    # phases[0].iter_nr=iter_start_curv+4000

    #owl
    # model.load_state_dict(torch.load("/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/checkpoints/s_384_Default_DEcayCurv50k_RGBWithSlicedFeat_DecayRGBmlpAndLatWD0.0_LatWd1.0_dtu_scan122/50000/models/sdf_model.pt") )
    # model_rgb.load_state_dict(torch.load("/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/checkpoints/s_384_Default_DEcayCurv50k_RGBWithSlicedFeat_DecayRGBmlpAndLatWD0.0_LatWd1.0_dtu_scan122/50000/models/rgb_model.pt") )
    # model_bg.load_state_dict(torch.load("/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/checkpoints/s_384_Default_DEcayCurv50k_RGBWithSlicedFeat_DecayRGBmlpAndLatWD0.0_LatWd1.0_dtu_scan122/50000/models/nerf_hash_bg_model.pt") )
    # model_colorcal.load_state_dict(torch.load("/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/checkpoints/s_384_Default_DEcayCurv50k_RGBWithSlicedFeat_DecayRGBmlpAndLatWD0.0_LatWd1.0_dtu_scan122/50000/models/colorcal_model.pt") )
    # phases[0].iter_nr=iter_start_curv+4000

    #fruit
    # model.load_state_dict(torch.load("/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/checkpoints/s_366_Default_DEcayCurv50k_dtu_scan63/50000/models/sdf_model.pt") )
    # model_rgb.load_state_dict(torch.load("/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/checkpoints/s_366_Default_DEcayCurv50k_dtu_scan63/50000/models/rgb_model.pt") )
    # model_bg.load_state_dict(torch.load("/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/checkpoints/s_366_Default_DEcayCurv50k_dtu_scan63/50000/models/nerf_hash_bg_model.pt") )
    # model_colorcal.load_state_dict(torch.load("/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/checkpoints/s_366_Default_DEcayCurv50k_dtu_scan63/50000/models/colorcal_model.pt") )
    # phases[0].iter_nr=iter_start_curv+4000

    #skull
    # model.load_state_dict(torch.load("/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/checkpoints/s_366_Default_DEcayCurv50k_dtu_scan65/50000/models/sdf_model.pt") )
    # model_rgb.load_state_dict(torch.load("/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/checkpoints/s_366_Default_DEcayCurv50k_dtu_scan65/50000/models/rgb_model.pt") )
    # model_bg.load_state_dict(torch.load("/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/checkpoints/s_366_Default_DEcayCurv50k_dtu_scan65/50000/models/nerf_hash_bg_model.pt") )
    # model_colorcal.load_state_dict(torch.load("/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/checkpoints/s_366_Default_DEcayCurv50k_dtu_scan65/50000/models/colorcal_model.pt") )
    # phases[0].iter_nr=iter_start_curv+4000

    #cams
    # model.load_state_dict(torch.load("/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/checkpoints/s_386_Default_DEcayCurv50k_RGBWithSlicedFeat_DecayRGBmlpAndLatWD0.0_LatWd1.0_curvDecayTo0_over1k_dtu_scan97/50000/models/sdf_model.pt") )
    # model_rgb.load_state_dict(torch.load("/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/checkpoints/s_386_Default_DEcayCurv50k_RGBWithSlicedFeat_DecayRGBmlpAndLatWD0.0_LatWd1.0_curvDecayTo0_over1k_dtu_scan97/50000/models/rgb_model.pt") )
    # model_bg.load_state_dict(torch.load("/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/checkpoints/s_386_Default_DEcayCurv50k_RGBWithSlicedFeat_DecayRGBmlpAndLatWD0.0_LatWd1.0_curvDecayTo0_over1k_dtu_scan97/50000/models/nerf_hash_bg_model.pt") )
    # model_colorcal.load_state_dict(torch.load("/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/checkpoints/s_386_Default_DEcayCurv50k_RGBWithSlicedFeat_DecayRGBmlpAndLatWD0.0_LatWd1.0_curvDecayTo0_over1k_dtu_scan97/50000/models/colorcal_model.pt") )
    # phases[0].iter_nr=iter_start_curv+4000


    ##crahes when compting importance samples
    # model.load_state_dict(torch.load("/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/checkpoints/s_389_trying_to_find_bug_with_impSampling_bmvs_bear/5000/models/sdf_model.pt") )
    # model_rgb.load_state_dict(torch.load("/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/checkpoints/s_389_trying_to_find_bug_with_impSampling_bmvs_bear/5000/models/rgb_model.pt") )
    # model_bg.load_state_dict(torch.load("/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/checkpoints/s_389_trying_to_find_bug_with_impSampling_bmvs_bear/5000/models/nerf_hash_bg_model.pt") )
    # model_colorcal.load_state_dict(torch.load("/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/checkpoints/s_389_trying_to_find_bug_with_impSampling_bmvs_bear/5000/models/colorcal_model.pt") )
    # phases[0].iter_nr=5000+4000



    while is_in_training_loop:

        for phase in phases:
            cb.epoch_started(phase=phase)
            cb.phase_started(phase=phase)

            while ( phase.samples_processed_this_epoch< phase.loader.nr_samples() ): #we assume we have the data
                if True: 
                    is_training = phase.grad
                    model.train(phase.grad)
                    model_rgb.train(phase.grad)
                    model_bg.train(phase.grad)


                    # #forward
                    with torch.set_grad_enabled(is_training):
                        cb.before_forward_pass() #sets the appropriate sigma for the lattice

                        # print("\n phase.iter_nr",  phase.iter_nr, "phase grad is ", phase.grad)

                        loss, loss_rgb, loss_eikonal, loss_curvature=init_losses() 

                        iter_nr_for_anneal=get_iter_for_anneal(phases[0].iter_nr, args.nr_iter_sphere_fit)
                        in_process_of_sphere_init=phases[0].iter_nr<args.nr_iter_sphere_fit
                        just_finished_sphere_fit=phases[0].iter_nr==args.nr_iter_sphere_fit


                        if in_process_of_sphere_init:
                            # loss, loss_sdf, loss_eikonal= loss_sphere_init(dataset_name, 1000, aabb, model, lattice, iter_nr_for_anneal, use_only_dense_grid )
                            loss, loss_sdf, loss_eikonal= loss_sphere_init(dataset_name, 30000, aabb, model, lattice, iter_nr_for_anneal, use_only_dense_grid=False )
                            # exit(1)
                        else:

                            # if phase.iter_nr%8==0:
                            #     with torch.set_grad_enabled(False):
                            #         # TIME_START("update_grid")
                            #         grid_centers_random, grid_center_indices=occupancy_grid.compute_random_sample_of_grid_points(256*256,True)
                            #         # TIME_END("get_centers_rand")
                            #         # show_points(grid_centers_random,"grid_centers_random")

                            #         # print('grid_centers',grid_centers)


                            #         # #get rgba field for all the centers
                            #         # density_field=model.get_only_density( grid_centers, lattice, phase.iter_nr) 
                            #         sdf,_,_=model( grid_centers_random, lattice, phase.iter_nr, False) 

                            #         # # print("density_field",density_field)

                            #         # #show grid centers
                            #         # grid_centers=occupancy_grid.compute_grid_points(False)
                            #         # grid_centers_eig=tensor2eigen(grid_centers)
                            #         # mesh_centers=Mesh()
                            #         # mesh_centers.V=grid_centers_eig
                            #         # # mesh_centers.C=tensor2eigen(color_by_idx(occupancy_grid.get_nr_voxels()))
                            #         # # mesh_centers.C=tensor2eigen(color_by_density_from_occupancy_grid(occupancy_grid))
                            #         # mesh_centers.C=tensor2eigen(color_by_occupancy_from_occupancy_grid(occupancy_grid))
                            #         # # mesh_centers.C=tensor2eigen(color_by_density(density_field))
                            #         # mesh_centers.m_vis.m_show_points=True
                            #         # mesh_centers.m_vis.set_color_pervertcolor()
                            #         # Scene.show(mesh_centers,"mesh_centers")

                            #         # print("sdf min max is ", sdf.min(), sdf.max())

                            #         #update the occupancy
                            #         # occupancy_grid.update_with_density(density_field, 0.95, 1e-3)
                            #         inv_s=math.exp(0.7*10)
                            #         max_eikonal_abs=0.0
                            #         occupancy_thresh=1e-6
                            #         occupancy_grid.update_with_sdf_random_sample(grid_center_indices, sdf, inv_s, max_eikonal_abs, occupancy_thresh )
                            #         # TIME_END("update_grid")



                            #sphere trace until the end and predict rgb
                            # TIME_START("rgb_prep")
                            with torch.set_grad_enabled(False):

                                ray_origins, ray_dirs, gt_selected, gt_mask_selected, img_indices=InstantNGP.random_rays_from_reel(tensor_reel, nr_rays_to_create)
                                ray_points_entry, ray_t_entry, ray_points_exit, ray_t_exit, does_ray_intersect_box=aabb.ray_intersection(ray_origins, ray_dirs)

                                ray_samples_packed=occupancy_grid.compute_samples_in_occupied_regions(ray_origins, ray_dirs, ray_t_entry, ray_t_exit, min_dist_between_samples, max_nr_samples_per_ray, model.training)
                                ray_samples_packed=ray_samples_packed.get_valid_samples()

                                ####IMPORTANCE sampling
                                # TIME_START("imp_sampling")
                                # inv_s_imp_sampling=1e4
                                do_imp_sampling=False
                                if do_imp_sampling:
                                    inv_s_imp_sampling=512
                                    inv_s_multiplier=1
                                    sdf_sampled_packed, _, _=model(ray_samples_packed.samples_pos, lattice, iter_nr_for_anneal, use_only_dense_grid=False)
                                    ray_samples_packed.set_sdf(sdf_sampled_packed) ##set sdf
                                    alpha=VolumeRendering.sdf2alpha(ray_samples_packed, sdf_sampled_packed, inv_s_imp_sampling, True, inv_s_multiplier)
                                    transmittance, bg_transmittance= VolumeRendering.cumprod_alpha2transmittance(ray_samples_packed, 1-alpha)
                                    weights = alpha * transmittance
                                    weights_sum, weight_sum_per_sample=VolumeRendering.sum_over_each_ray(ray_samples_packed, weights)
                                    weight_sum_per_sample[weight_sum_per_sample==0]=1e-6 #prevent nans
                                    weights/=weight_sum_per_sample #prevent nans
                                    cdf=VolumeRendering.compute_cdf(ray_samples_packed, weights)
                                    ray_samples_packed_imp=VolumeRendering.importance_sample(ray_origins, ray_dirs, ray_samples_packed, cdf, 16, model.training)
                                    sdf_sampled_packed_imp, _, _=model(ray_samples_packed_imp.samples_pos, lattice, iter_nr_for_anneal, use_only_dense_grid=False)
                                    ray_samples_packed_imp.set_sdf(sdf_sampled_packed_imp) ##set sdf
                                    ray_samples_combined=VolumeRendering.combine_uniform_samples_with_imp(ray_origins, ray_dirs, ray_samples_packed, ray_samples_packed_imp)
                                    ray_samples_packed=ray_samples_combined#swap
                                    ray_samples_packed=ray_samples_packed.get_valid_samples() #still need to get the valid ones because we have less samples than allocated
                                    ####SECOND ITER
                                    inv_s_multiplier=2
                                    sdf_sampled_packed=ray_samples_packed.samples_sdf #we already combined them and have the sdf
                                    alpha=VolumeRendering.sdf2alpha(ray_samples_packed, sdf_sampled_packed, inv_s_imp_sampling, True, inv_s_multiplier)
                                    transmittance, bg_transmittance= VolumeRendering.cumprod_alpha2transmittance(ray_samples_packed, 1-alpha)
                                    weights = alpha * transmittance
                                    weights_sum, weight_sum_per_sample=VolumeRendering.sum_over_each_ray(ray_samples_packed, weights)
                                    weight_sum_per_sample[weight_sum_per_sample==0]=1e-6 #prevent nans
                                    weights/=weight_sum_per_sample #prevent nans
                                    cdf=VolumeRendering.compute_cdf(ray_samples_packed, weights)
                                    ray_samples_packed_imp=VolumeRendering.importance_sample(ray_origins, ray_dirs, ray_samples_packed, cdf, 16, model.training)
                                    ray_samples_combined=VolumeRendering.combine_uniform_samples_with_imp(ray_origins, ray_dirs, ray_samples_packed, ray_samples_packed_imp)
                                    ray_samples_packed=ray_samples_combined#swap
                                    ray_samples_packed=ray_samples_packed.get_valid_samples() #still need to get the valid ones because we have less samples than allocated
                                    show_points(ray_samples_packed_imp.samples_pos,"samples_pos_imp")
                                # print("ray_samples_combined.curnrsamples",ray_samples_combined.cur_nr_samples)
                                # print("ray_samples_combined.max_nr_samples",ray_samples_combined.max_nr_samples)
                                # TIME_END("imp_sampling")
                                #####FINISH imp sampling
                                show_points(ray_samples_packed.samples_pos,"samples_pos")
                                # show_points(ray_samples_packed.samples_pos,"samples_pos", color_per_vert=weights.view(-1,1).repeat(1,3))
                                # show_points(ray_samples_packed.samples_pos,"samples_pos", color_per_vert=cdf.view(-1,1).repeat(1,3))
                                # imp_remade=ray_origins+ray_samples_packed_imp.samples_z*ray_dirs
                                # show_points(imp_remade,"imp_remade")


                                # #make ray samples
                                # TIME_START("ray_sample")
                                z_vals, z_vals_imp = model.ray_sampler.get_z_vals(ray_origins, ray_dirs, model, lattice, iter_nr_for_anneal, use_only_dense_grid=False) #nr_rays x nr_samples
                                # TIME_END("ray_sample")

                                #get mid points
                                z_vals_rgb = get_midpoint_of_sections(z_vals) #gets a z value for each midpoint 

                                
                                ray_samples_sdf = ray_origins[:, None, :] + ray_dirs[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
                                ray_samples_rgb = ray_origins[:, None, :] + ray_dirs[:, None, :] * z_vals_rgb[..., :, None]  # n_rays, n_samples, 3
                                dirs = ray_dirs[:, None, :].expand(ray_samples_rgb.shape )
                                nr_rays=ray_samples_rgb.shape[0]
                                nr_samples=ray_samples_rgb.shape[1]
                                # print("dirsis ", dirs.shape)


                                #new stuff based on neus
                                # pts_sdf = ray_samples_sdf.reshape(-1, 3)
                                pts = ray_samples_rgb.reshape(-1, 3)
                                dirs = dirs.reshape(-1, 3)


                                if args.without_mask:
                                    z_vals_bg, dummy, ray_samples_bg_4d, ray_samples_bg = model_bg.ray_sampler_bg.get_z_vals_bg(ray_origins, ray_dirs, model_bg, lattice_bg, iter_nr_for_anneal)
                                    dirs_bg = ray_dirs[:, None, :].expand(ray_samples_bg.shape ).contiguous()

                                    ray_samples_packed_bg= RaySampler.compute_samples_bg(ray_origins, ray_dirs, ray_t_exit, nr_samples_bg, aabb.m_radius, aabb.m_center_tensor, model.training)

                            # TIME_END("rgb_prep")

                            # #predict sdf
                            # TIME_START("sdf_and_rgb")
                            sdf, sdf_gradients, feat, _=model.get_sdf_and_gradient(pts, lattice, iter_nr_for_anneal, use_only_dense_grid=False)

                            # print("spatial variance min max", variance.min(), variance.max())




                            # #predict rgb
                            rgb_samples, rgb_samples_view_dep = model_rgb(model, feat, sdf_gradients, pts, dirs, lattice, iter_nr_for_anneal, model_colorcal, img_indices)
                            rgb_samples=rgb_samples.view(nr_rays, -1, 3)
                            # TIME_END("sdf_and_rgb")

                            #volume render
                            cos_anneal_ratio=map_range_val(iter_nr_for_anneal, 0.0, forced_variance_finish_iter, 0.0, 1.0)
                            forced_variance=map_range_val(iter_nr_for_anneal, 0.0, forced_variance_finish_iter*1, 0.3, 0.8) #DO NOT go above 0.8 or so or the gradient gets too shart and when we lower the curvature we don't recover much detail anymore like the screw in the scissors dataset
                            #IMPORTANT to make the forced variance kinda slowly increasing, it should reach 0.7 at around 30k iteration. Anything faster and the stem of the fruit dissapears
                            # linear_iter_01=map_range_val(iter_nr_for_anneal, 0.0, forced_variance_finish_iter*1, 0.0, 1.0)
                            # linear_iter_01_smoothstop=smoothstop_n(linear_iter_01,3)
                            # forced_variance=map_range_val(linear_iter_01_smoothstop, 0.0, 1.0, 0.3, 0.8)
                            # forced_variance=None
                            # TIME_START("vol_render")
                            weights, inv_s, inv_s_before_exp, bg_transmittance = model.volume_renderer(pts, lattice, z_vals, ray_t_exit, sdf, sdf_gradients, dirs, nr_rays, nr_samples, cos_anneal_ratio, forced_variance=forced_variance) #neus
                            # TIME_END("vol_render")
                            # weights, inv_s, inv_s_before_exp, bg_transmittance = model.volume_renderer.forward2(sdf, nr_rays, nr_samples, forced_variance=forced_variance) #neus
                            pred_rgb = torch.sum(weights.unsqueeze(-1) * rgb_samples, 1)


                            #####make the samples pack to be the same as the neus samples
                            # use_samples_from_neus=False
                            # if use_samples_from_neus:
                            #     ray_samples_packed=RaySamplesPacked(nr_rays, nr_rays*nr_samples)
                            #     ray_samples_packed.rays_have_equal_nr_of_samples=True
                            #     ray_samples_packed.fixed_nr_of_samples_per_ray=nr_samples
                            #     ray_samples_packed.samples_pos=pts.view(-1,3)
                            #     ray_samples_packed.samples_dirs=dirs.view(-1,3)
                            #     ray_samples_packed.samples_z=z_vals_rgb.view(-1,1)
                            #     dists = z_vals_rgb[:, 1:] - z_vals_rgb[:, :-1]
                            #     dists = torch.cat([dists, ray_t_exit - z_vals_rgb[:, -1:]], -1) # included also the dist from the sphere intersection
                            #     ray_samples_packed.samples_dt=dists.view(-1,1)


                            # print("ray_samples_packed.samples_pos min max ", ray_samples_packed.samples_pos.min(), ray_samples_packed.samples_pos.max())
                            #######################FUSED stuff
                            # TIME_START("sdf_and_rgb_fused")
                            ###FUSED sdf--------------------
                            sdf_fused, sdf_gradients_fused, feat_fused, _ =model.get_sdf_and_gradient(ray_samples_packed.samples_pos, lattice, iter_nr_for_anneal, use_only_dense_grid=False)
                            #FUSED vol render----------------------
                            # print("vol_render")
                            weights_fused, weights_sum_fused, inv_s_fused, inv_s_before_exp_fused, bg_transmittance_fused = model.volume_renderer.vol_render_samples_packed(ray_samples_packed, ray_t_exit, True, sdf_fused, sdf_gradients_fused, cos_anneal_ratio, forced_variance=forced_variance) #neus
                            # print("finished_vol_render")
                            # print("weights_fused min max is ", weights_fused.min(), weights_fused.max())
                            # print("weights_fused mean is ", weights_fused.mean())
                            # print("weights mean is ", weights.mean())
                            #Fused RGB---------------------------
                            rgb_samples_fused, _ = model_rgb(model, feat_fused, sdf_gradients_fused, ray_samples_packed.samples_pos, ray_samples_packed.samples_dirs, lattice, iter_nr_for_anneal, model_colorcal, img_indices, ray_samples_packed.ray_start_end_idx)
                            # print("rgb_samples_fused min max is ", rgb_samples_fused.min(), rgb_samples_fused.max())
                            #FUSED integrate weigths and rgb_samples_fused
                            # pred_rgb_fused=VolumeRendering.integrate_rgb_and_weights(ray_samples_packed, rgb_samples_fused, weights_fused)
                            pred_rgb_fused=model.volume_renderer.integrator_module(ray_samples_packed, rgb_samples_fused, weights_fused)
                            #try again but with multiply and sum in two different ones
                            # rgb_samples_weighted=rgb_samples_fused*weights_fused
                            # pred_rgb_fused, _=model.volume_renderer.sum_ray_module(ray_samples_packed, rgb_samples_weighted)
                            # TIME_END("sdf_and_rgb_fused")
                            
                            #diff
                            # diff=((pred_rgb-pred_rgb_fused)**2).mean()
                            # print("difff is ",diff)
                            #####################fused stuff
                            # print("exit after one iteration")
                            # exit(1)
                            #switch to the fused one so we optimize this one
                            optimize_fused=False
                            if optimize_fused:
                                pred_rgb=pred_rgb_fused
                                sdf=sdf_fused
                                sdf_gradients=sdf_gradients_fused
                                pts=ray_samples_packed.samples_pos
                                weights=weights_fused
                                weights_sum=weights_sum_fused
                                bg_transmittance=bg_transmittance_fused
                            else:
                                weights_sum=torch.sum(weights.unsqueeze(-1) , 1)



                            #run nerf bg
                            if args.without_mask:
                                rgb_samples_bg, density_samples_bg=model_bg( ray_samples_bg_4d.view(-1,4), dirs_bg.view(-1,3), lattice_bg, iter_nr_for_anneal, model_colorcal, img_indices) 
                                rgb_samples_bg=rgb_samples_bg.view(nr_rays_to_create, nr_samples_bg,3)
                                density_samples_bg=density_samples_bg.view(nr_rays_to_create, nr_samples_bg)
                                # #get weights for the integration
                                weights_bg, disp_map_bg, acc_map_bg, depth_map_bg, _=model_bg.volume_renderer(density_samples_bg, z_vals_bg, None)
                                pred_rgb_bg = torch.sum(weights_bg.unsqueeze(-1) * rgb_samples_bg, 1)


                                # rgb_samples_bg, density_samples_bg=model_bg( ray_samples_packed_bg.samples_pos_4d.view(-1,4), ray_samples_packed_bg.samples_dirs.view(-1,3), lattice_bg, iter_nr_for_anneal, model_colorcal, img_indices, nr_rays=nr_rays_to_create) 
                                # rgb_samples_bg=rgb_samples_bg.view(nr_rays_to_create, nr_samples_bg,3)
                                # density_samples_bg=density_samples_bg.view(nr_rays_to_create, nr_samples_bg)
                                # # #get weights for the integration
                                # # weights_bg, disp_map_bg, acc_map_bg, depth_map_bg, _=model_bg.volume_renderer(density_samples_bg, z_vals_bg, None)
                                # pred_rgb_bg, pred_depth_bg, _= model_bg.volume_renderer_general.volume_render_nerf(ray_samples_packed_bg, rgb_samples_bg.view(-1,3), density_samples_bg.view(-1,1), ray_t_exit, False)
                         

                                #combine attempt 3 like in https://github.com/lioryariv/volsdf/blob/a974c883eb70af666d8b4374e771d76930c806f3/code/model/network_bg.py#L96
                                pred_rgb_bg = bg_transmittance.view(-1,1) * pred_rgb_bg
                                pred_rgb = pred_rgb + pred_rgb_bg
                            

                            #calib rgb
                            # pred_rgb=model_rgb.model_color_calib.calib_RGB_lin(pred_rgb, str(frame.frame_idx) )

                            # pred_rgb=model_colorcal.calib_RGB_nchw(pred_rgb, 1)
                            # pred_rgb=model_colorcal.calib_RGB_lin_reel(pred_rgb, img_indices)


                            if phase.iter_nr%8==0:
                                with torch.set_grad_enabled(False):
                                    grid_centers_random, grid_center_indices=occupancy_grid.compute_random_sample_of_grid_points(256*256*4,True)
                                    sdf_grid,_,_=model( grid_centers_random, lattice, iter_nr_for_anneal, False) 
                                    inv_s=inv_s_fused.item()
                                    grad_norm=torch.linalg.norm(sdf_gradients_fused.detach().reshape(-1, 3), ord=2, dim=-1)
                                    max_eikonal_abs= (grad_norm - 1.0).abs().max()
                                    # print("max_eikonal_abs",max_eikonal_abs)
                                    # print("min max grad norm", grad_norm.min(), grad_norm.max())
                                    # print("ray_dixed dt min max is ",ray_samples_packed.ray_fixed_dt.min(), ray_samples_packed.ray_fixed_dt.max())
                                    # ray_fixed_dt_different_than=ray_samples_packed.ray_fixed_dt!=0
                                    # print("ray_dixed dt different0 min max is ",ray_samples_packed.ray_fixed_dt[ray_fixed_dt_different_than].min(), ray_samples_packed.ray_fixed_dt[ray_fixed_dt_different_than].max())
                                    occupancy_thresh=1e-5
                                    occupancy_grid.update_with_sdf_random_sample(grid_center_indices, sdf_grid, inv_s, max_eikonal_abs, occupancy_thresh )

                                    #adjust nr_rays_to_create based on how many samples we have in total
                                    cur_nr_samples=ray_samples_packed.samples_pos.shape[0]
                                    multiplier_nr_samples=float(target_nr_of_samples)/cur_nr_samples
                                    nr_rays_to_create=int(nr_rays_to_create*multiplier_nr_samples)
                                    


                            #rgb loss
                            loss_rgb=rgb_loss(gt_selected, pred_rgb, does_ray_intersect_box)
                            loss+=loss_rgb

                            #eikonal loss
                            loss_eikonal =eikonal_loss(sdf_gradients)
                            loss+=loss_eikonal*args.eik_w #10 here and 30 in the curvatures makes the dog looks ncie but loses paws


                            #loss that makes the vairance kinda close uniformly
                            # variance_close=inv_s_before_exp.std()
                            # print("variance_close",variance_close)
                            # loss+=variance_close*0.1

                            # if sdf_residual is not None:
                            #     distance_allowed_without_penalty=1/model.dense_grid_size
                            #     distance_allowed_without_penalty=distance_allowed_without_penalty/2
                            #     residuals_too_small=sdf_residual.abs()<distance_allowed_without_penalty
                            #     loss_residual_small=(sdf_residual**2)
                            #     loss_residual_small[residuals_too_small]=0.0
                            #     loss_residual_small=loss_residual_small.mean()
                            #     print("loss_residual_small",loss_residual_small)
                            #     print("sdf residual abs avg, max", sdf_residual.abs().mean(), sdf_residual.abs().max())
                            #     # print("CURRENTLY it is disabled")
                            #     loss+=loss_residual_small*1e1


                            #loss on curvature
                            use_loss_curvature=True
                            loss_curvature=torch.tensor(0)
                            if use_loss_curvature:
                                sdf2, sdf_curvature, feat2, _=model.get_sdf_and_curvature_1d_precomputed_gradient( pts, sdf, sdf_gradients, lattice, iter_nr_for_anneal, use_only_dense_grid=False)
                                # sdf2, sdf_curvature, feat2=model.get_sdf_and_curvature_1d_precomputed_gradient_normal_based( pts, sdf_gradients, lattice, iter_nr_for_anneal)
                                # sdf_curvature=sdf_curvature.view(nr_rays, -1)
                                
                                # loss_curvature=(sdf_curvature.abs() *weights.view(nr_rays,-1).detach()  ) 
                                loss_curvature=(sdf_curvature.abs().view(-1) *weights.view(-1).detach()  ) 
                                # loss_curvature=(sdf_curvature.abs().view(-1)   )  #if we use occupancy and multiply weight weight, we start with low W for the curvature a the surface and increase it later but by then it's already too late to recover from the shitty curvature we have now
                                # loss_curvature=torch.ones_like(loss_curvature)
                                # loss_curvature=loss_curvature*
                                # loss_curvature=robust_loss_pytorch.general.lossfun(loss_curvature, alpha=torch.tensor(-2.0), scale=torch.tensor(0.1))
                                # print("loss_curvature abs dot is ", loss_curvature.min(), loss_curvature.max())
                                # print("sdf_curvatured min max is ", sdf_curvature.min(), sdf_curvature.max())
                                loss_curvature=loss_curvature.mean()
                                global_weight_curvature=map_range_val(iter_nr_for_anneal, iter_start_curv, iter_finish_curv, 1.0, 0.000) #once we are converged onto good geometry we can safely descrease it's weight so we learn also high frequency detail geometry.
                                loss+=loss_curvature* args.curv_w*1e-3 *global_weight_curvature #THIS should maybe promote the curvature tangential to the normal because in the direction of the normal it just promotes a constant sdf


                            


                            use_loss_grad=False
                            if use_loss_grad:
                                loss_sdf_grad=sdf_grad_rand.abs().mean()
                                loss+=loss_sdf_grad*1e-1

                            #loss that says the sdf should be high
                            use_loss_high_sdf=True
                            if use_loss_high_sdf and iter_nr_for_anneal>=iter_start_curv and isinstance(loader_train, DataLoaderPhenorobCP1):
                                offsurface_points=aabb_big.rand_points_inside(nr_points=3*60000)
                                sdf_rand, feat_rand=model( offsurface_points, lattice, iter_nr_for_anneal)
                                # loss_high_sdf=( torch.nn.functional.softplus(-sdf_rand, beta=30.0)  ).mean()
                                loss_high_sdf=( torch.nn.functional.relu(-sdf_rand)  ).mean() #do a relu because doing a softplus also penalizes the psitiive side of the sdf which makes it closer to 0 therefore it gets more weight and the whole recontruction looks blurrier
                                loss+=loss_high_sdf*1e-2
                                # loss+=loss_high_sdf*1

                            #highsdf just to avoice voxels becoming "occcupied" due to their sdf dropping to zero
                            offsurface_points=aabb_big.rand_points_inside(nr_points=512)
                            sdf_rand, _, _=model( offsurface_points, lattice, iter_nr_for_anneal, use_only_dense_grid=False)
                            loss_offsurface_high_sdf=torch.exp(-1e2 * torch.abs(sdf_rand)).mean()
                            # print("loss_offsurface_high_sdf",loss_offsurface_high_sdf)
                            loss+=loss_offsurface_high_sdf*1e-3


                            #use alpha prior to be a beta distribution similar to neural volumes
                            #It seems to introduce waviness
                            # if not use_mask_loss:
                            #     rayalpha=weights.sum(dim=1, keepdim=True)
                            #     # rayalpha=bg_transmittance
                            #     alphaprior = torch.mean(
                            #         torch.log(0.1 + rayalpha.view(rayalpha.size(0), -1)) +
                            #         torch.log(0.1 + 1. - rayalpha.view(rayalpha.size(0), -1)) - -2.20727, dim=-1)
                            #     if(iter_nr_for_anneal>forced_variance_finish_iter):
                            #         loss+=alphaprior.mean()*0.1


                            #any points that are not supervized by the gradient error should just have a constant value because it's the easiest to learn
                            #the value should be 3*eik_clamp because that is where 99% of the loss of the gradient error is gone and we are therefore outside of the band that is supervized
                            use_constant_sdf_loss=False
                            if use_constant_sdf_loss:
                                eik_clamp=0.05
                                x=sdf_rand.abs().detach()
                                c=eik_clamp #standard_deviation
                                w_constantsdf=1.0 - torch.exp(- (x*x)/(2*c*c)    ) #We use a guassian, for high sdf we have a lower eik loss because we are not so interesting in it being perfect when far away from the surface
                                loss_constant_sdf=(3*eik_clamp-torch.clamp(sdf_rand,min=0.0)) #if its negative don't supervize it, if it positive make it 3eikcamp
                                loss_constant_sdf=loss_constant_sdf.clamp(min=0.0) #makes so that any value below 3*eikclamp is supervized by any sdf above is left without a loss to be free to be whatever
                                loss_constant_sdf=loss_constant_sdf * w_constantsdf
                                loss_constant_sdf=loss_constant_sdf.mean()
                                loss+=loss_constant_sdf*10 ###if you disable this, you also need to disable the clamping of the eik because otherwise most sdf will be close to zero and then eik loss takes effect

                            if use_mask_loss:
                                # weights_sum=torch.sum(weights.unsqueeze(-1) , 1)
                                # print("weights_sum min max is ", weights_sum.min(), weights_sum.max())
                                loss_mask=torch.nn.functional.binary_cross_entropy(weights_sum.clip(1e-3, 1.0 - 1e-3), gt_mask_selected)
                                loss+=loss_mask*0.1 #0.1 works fine for getting the drill bit of the scissors dataset, anything lower is worse


                            # print("loss is ", loss)

                            #make the trainable iter high
                            # loss+=(1-model.trainable_iter.abs()).abs().mean()*0.001 
                            # print("model.trainable_iter",model.trainable_iter)

                            # print("phase.iter_nr",phase.iter_nr)
                            # print("rgb_samples",rgb_samples.mean())
                            # print("weights",weights.mean())
                            # print("rgb_samples_bg",rgb_samples_bg.mean())
                            # print("weights_bg",weights_bg.mean())
                            # print("ray_samples_bg_4d",ray_samples_bg_4d.mean())
                            # print("pred_rgb", pred_rgb.mean())
                            # print("pred_rgb_bg", pred_rgb_bg.mean())
                            # print("loss_rgb", loss_rgb)
                            # print("loss_eikonal", loss_eikonal)
                            # print("alphaprior.mean()", alphaprior.mean())                            


                            if torch.isnan(loss).any():
                                print("detected nan in loss at iter", phase.iter_nr)
                                # print("pred_rgb", pred_rgb.mean())
                                # print("pred_rgb_bg", pred_rgb_bg.mean())
                                # print("loss_rgb", loss_rgb)
                                # print("loss_eikonal", loss_eikonal)
                                # print("alphaprior.mean()", alphaprior.mean())
                                exit()

                            



                        # if phase.iter_nr%1000==0 and is_training and with_viewer and not in_process_of_sphere_init:
                            # show_points(ray_samples,"ray_samples", color_per_vert=weights.view(-1,1).repeat(1,3)  )
                            # show_points(ray_samples,"ray_samples"  )
                            # weight_color=weights.view(-1,1).repeat(1,3)
                            # show_points(ray_samples_rgb,"ray_samples_rgb", color_per_vert=weight_color   )

                            #show the points with high weight
                            # high_weight_mask=weights.view(-1,1)>0.01
                            # weights_high_selected=weights.view(-1,1)[high_weight_mask].view(-1,1)
                            # points_high_weight=ray_samples_rgb.view(-1,3)[high_weight_mask.repeat(1,3)].view(-1,3)
                            # print("weights_high_selected",weights_high_selected.shape)
                            # print("points_high_weight",points_high_weight.shape)
                            # show_points(points_high_weight,"points_high_weight", color_per_vert=weights_high_selected.view(-1,1).repeat(1,3)   )


                            #get also the last 16 importance samples
                            # ray_samples_rgb_final_imp=ray_samples_rgb.view(nr_rays,-1,3)
                            # nr_samples_last_imp=model.N_samples_importance//model.n_iters_upsample
                            # ray_samples_rgb_final_imp=ray_samples_rgb_final_imp[:,-nr_samples_last_imp:,:]
                            # ray_samples_rgb_final_imp=ray_samples_rgb_final_imp.reshape(-1,3)
                            # show_points(ray_samples_rgb_final_imp,"ray_samples_rgb_final_imp"  )

                            #last 16 importance samples
                            # show_points(ray_samples_imp,"ray_samples_imp"  )

                        


                        #TODO make just a normal nerf renderer using the instant ngp encoding
                        #TODO make an adaptive pixel sampler such that we sample the pixels with high error with more probability. this will make the tiny errors near the ear hopefully dissaper
                        #TODO maybe the problem with the breathing and the convergence is due tot he fact that the capped points are not assined an sdf of zero at the border? Maybe that will help to concentrate some radiance there. 
                        #TODO implement the slicing from dense grid up until the nr of elements is smaller than the lattice nr of vertices
                        #TODO why the fuck is the mesh breathing??
                        #TODO maybe do the eikonal loss only on the samples and not on offsurface points?
                        #TODO finer lattice at o.0001 seems to help the geomtry
                        #TODO get also the normal of the samples and give it to the rgb_model, or better yet, make the sdf, also predict a small feature that is then used by the rgb model
                        #TODO make the rendering code more reusable so that in the evaluation we can also use the same settings
                        #TODO make it so that if we have no low sdf, and the weights are all high, then we set the color to black or white
                        #TODO sample RGB also at random points from ray origin to ray end, weigthed by the SDF weight. This will make it put gradient in ares that might need to be newly surfaced.
                        #TODO slowly reduce the rance of the sdf weight during optimization so that we start with the whole volume with a weight of 1 everywhere and then slowly move towards only the surface
                        #TODO loss on the surface of the rendered normal map, this might help fill in the gaps in the surface






                        #if its the first time we do a forward on the model we need to create here the optimizer because only now are all the tensors in the model instantiated
                        # lr=train_params.lr()
                        if first_time or just_finished_sphere_fit or iter_nr_for_anneal==iter_start_curv:
                        # if first_time or just_finished_sphere_fit:
                        # if first_time:
                            first_time=False

                            #if we finished with optimizing curvature, we reduce lr, recreate the rgb and keep optimizing with a lower curvature loww
                            # if iter_nr_for_anneal==iter_start_curv:
                                # model_rgb=RGB(nr_lattice_vertices, nr_lattice_features, nr_resolutions=nr_resolutions, feat_size_in=32, nr_cams=loader_train.nr_samples() ).to("cuda")
                                # print("recreated rgb model with lr", lr)


                            params=[]
                            model_params_without_lattice=[]
                            for name, param in model.named_parameters():
                                if "lattice_values" in name:
                                    pass
                                else:
                                    model_params_without_lattice.append(param)
                            model_rgb_params_without_lattice=[]
                            for name, param in model_rgb.named_parameters():
                                if "lattice_values" in name:
                                    pass
                                else:
                                    model_rgb_params_without_lattice.append(param)


                        

                            # params.append( {'params': model_params_without_lattice, 'weight_decay': train_params.weight_decay(), 'lr': train_params.lr()} )
                            # params.append( {'params': model_rgb_params_without_lattice, 'weight_decay': train_params.weight_decay(), 'lr': train_params.lr()} )
                            # params.append( {'params': model.lattice_values_monolithic, 'weight_decay': train_params.weight_decay(), 'lr': train_params.lr()*3} )
                            # params.append( {'params': model_rgb.lattice_values_monolithic, 'weight_decay': train_params.weight_decay(), 'lr': train_params.lr()*3} )


                            #also leaves the variance with high lr
                            # params.append( {'params': model.volume_renderer.deviation_network.variance, 'weight_decay': train_params.weight_decay(), 'lr': train_params.lr()*3.33} )
                            # params.append( {'params': model.mlp_sdf.parameters(), 'weight_decay': train_params.weight_decay(), 'lr': train_params.lr()} )
                            # params.append( {'params': model.lattice_values_monolithic, 'weight_decay': train_params.weight_decay(), 'lr': train_params.lr()*3.33} )
                            # params.append( {'params': model_rgb.model_color_calib.parameters(), 'weight_decay': train_params.weight_decay(), 'lr': train_params.lr()} )
                            # params.append( {'params': model_rgb.mlp.parameters(), 'weight_decay': train_params.weight_decay(), 'lr': train_params.lr()} )
                            # params.append( {'params': model_rgb.lattice_values_monolithic, 'weight_decay': train_params.weight_decay(), 'lr': train_params.lr()*3.33} )
                            # if not model.volume_renderer.single_variance:
                            #     params.append( {'params': model.volume_renderer.deviation_network.mlp.parameters(), 'weight_decay': train_params.weight_decay(), 'lr': train_params.lr()} )
                            #     params.append( {'params': model.volume_renderer.deviation_network.lattice_values_monolithic, 'weight_decay': train_params.weight_decay(), 'lr': train_params.lr()*3.33} )



                            #with weight decay for the mlp part of the rgb so that most of the color is not fake
                            # params.append( {'params': model.volume_renderer.deviation_network.variance, 'weight_decay': 0, 'lr': train_params.lr()} )
                            # params.append( {'params': model.mlp_sdf.parameters(), 'weight_decay': 0, 'lr': train_params.lr()} )
                            # params.append( {'params': model.lattice_values_monolithic, 'weight_decay': 0, 'lr': train_params.lr()} )
                            # params.append( {'params': model_rgb.model_color_calib.parameters(), 'weight_decay': 0, 'lr': train_params.lr()} )
                            # params.append( {'params': model_rgb.mlp.parameters(), 'weight_decay': 2e-3, 'lr': train_params.lr()} )
                            # params.append( {'params': model_rgb.lattice_values_monolithic, 'weight_decay': 0, 'lr': train_params.lr()} )
                            # if not model.volume_renderer.single_variance:
                            #     params.append( {'params': model.volume_renderer.deviation_network.mlp.parameters(), 'weight_decay': 0, 'lr': train_params.lr()} )
                            #     params.append( {'params': model.volume_renderer.deviation_network.lattice_values_monolithic, 'weight_decay': 0, 'lr': train_params.lr()} )


                            

                            # params.append( {'params': model.parameters(), 'weight_decay': train_params.weight_decay(), 'lr': train_params.lr()} )
                            # params.append( {'params': model_rgb.parameters(), 'weight_decay': train_params.weight_decay(), 'lr': train_params.lr()} )
                            params.append( {'params': model.parameters(), 'weight_decay': 0.0, 'lr': lr } )

                            # params.append( {'params': model_params_without_lattice, 'weight_decay': 0.0, 'lr': lr} )
                            # params.append( {'params': model.lattice_values_monolithic, 'weight_decay': 0.0, 'lr': 3e-3} )

                            # params.append( {'params': model_rgb.parameters(), 'weight_decay': 0.0, 'lr': lr } )
                            # apply more weight decay to the lattice values for rgb
                            if iter_nr_for_anneal==iter_start_curv:
                                params.append( {'params': model_rgb_params_without_lattice, 'weight_decay': 0.1, 'lr': lr} )
                                params.append( {'params': model_rgb.lattice_values_monolithic, 'weight_decay': 1.0, 'lr': lr} )
                            else:
                                params.append( {'params': model_rgb_params_without_lattice, 'weight_decay': 0.0, 'lr': lr} )
                                params.append( {'params': model_rgb.lattice_values_monolithic, 'weight_decay': 0.0, 'lr': lr} )

                            params.append( {'params': model_bg.parameters(), 'weight_decay': 0.0, 'lr': lr } )
                            params.append( {'params': model_colorcal.parameters(), 'weight_decay': 1e-1, 'lr': lr } )
                            # params.append( {'params': model_colorcal.parameters(), 'weight_decay': 0.0, 'lr': lr } )
                            #RGB mlp has weight decay
                            # params.append( {'params': model_rgb.lattice_values_monolithic, 'weight_decay': train_params.weight_decay(), 'lr': lr } )
                            # params.append( {'params': model_rgb.mlp.parameters(), 'weight_decay': 3e-2, 'lr': lr } )


                            # params.append( {'params': model_rgb.parameters(), 'weight_decay': 1e-2, 'lr': lr } )
                            # params.append( {'params': adaptive_loss.parameters(), 'weight_decay': train_params.weight_decay(), 'lr': lr } )

                            # wd_list=np.linspace(0.0, 0.00001, num=model.nr_resolutions)
                            # for i in range(model.nr_resolutions):
                                # params.append( {'params': model.lattice_values_list[i], 'weight_decay': wd_list[i], 'lr': train_params.lr()} )

                            # optimizer = torch.optim.AdamW (params, amsgrad=True)  
                            # optimizer = torch.optim.AdamW (params, amsgrad=False,  betas=(0.9, 0.99), eps=1e-15) #params from instantngp
                            # optimizer = torch.optim.AdamW (params, amsgrad=False)  #no shingles
                            # optimizer = LazyAdam(params, betas=(0.9, 0.99), eps=1e-15) #seems to be a bit faster than adam at the beggining
                            # optimizer = RAdam(params, betas=(0.9, 0.99), eps=1e-15) #makes it slightly "noiser" with more curvature but solves the waviness
                            optimizer=apex.optimizers.FusedAdam(params, adam_w_mode=True, betas=(0.9, 0.99), eps=1e-15)
                            # optimizer = RAdam(params) 
                            # optimizer = Adan (params)  #no shingles
                            # optimizer = torch.optim.Adamax (params)
                            # optimizer = torch.optim.Adamax (params, betas=(0.9, 0.99), eps=1e-15)
                            # scheduler = torch.optim.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=20000)
                            scheduler_warmup=None
                            if just_finished_sphere_fit:
                                scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=3000) #anneal over total_epoch iterations
                            # scheduler_lr_decay=torch.optim.lr_scheduler.StepLR(optimizer,step_size=5000, gamma=0.1)
                            # scheduler_lr_decay=LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=100000)
                            # scheduler_lr_decay=LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=forced_variance_finish_iter*2)
                            # scheduler_lr_decay= MultiStepLR(optimizer, milestones=[40000,80000,100000], gamma=0.3, verbose=False)
                            scheduler_lr_decay= MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.3, verbose=False)
                            # scheduler_lr_decay= torch.optim.lr_scheduler.ExponentialLR(optimizer,)

                            scaler = GradScaler()

                        # loss=loss*1e-5

                        cb.after_forward_pass(loss=loss.item(), loss_rgb=loss_rgb, loss_sdf_surface_area=0, loss_sdf_grad=0, phase=phase, loss_eikonal=loss_eikonal.item(), loss_curvature=loss_curvature.item(), neus_variance_mean=model.volume_renderer.deviation_network.get_variance_item(), lr=optimizer.param_groups[0]["lr"]) #visualizes the prediction 
                        # cb.after_forward_pass(loss=loss.item(), loss_rgb=loss_rgb_l1, loss_sdf_surface_area=0, loss_sdf_grad=0, phase=phase, gradient_error=gradient_error, loss_curvature=loss_curvature.item(), neus_variance_mean=0, lr=optimizer.param_groups[0]["lr"]) #visualizes the prediction 
                        # print("lr ", optimizer.param_groups[0]["lr"])

                        #update gui
                        if with_viewer:
                            # prev_c2f_progress=ngp_gui.m_c2f_progress
                            ngp_gui.m_c2f_progress=model.c2f.get_last_t()
                            # print("last t is ", model.c2f.get_last_t())
                            # print("prev_c2f_progress",prev_c2f_progress)
                            # diff=abs(model.c2f.get_last_t()-prev_c2f_progress)
                            # if( diff>0.1 and diff!=0.3 ):
                            #     print("wtf why did we have such a big jump")
                            #     exit(1)



                    #backward
                    if is_training:
                        optimizer.zero_grad()
                        cb.before_backward_pass()
                        TIME_START("backward")
                        loss.backward()
                        # scaler.scale(loss).backward()
                        TIME_END("backward")

                        if scheduler_warmup is not None:
                            scheduler_warmup.step(iter_nr_for_anneal)

                        # print("-------------------------------------------------\n")
                        # summary(model)
                        # summary(model_rgb)
                        # print("model last lattice min max", model.lattice_values_list[-1].min(), model.lattice_values_list[-1].max())
                        # print("model last lattice mean std", model.lattice_values_list[-1].mean(), model.lattice_values_list[-1].std())


                        #apply weight decay to the lattice values https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/8
                        # print("model.lattice_values_monolithic.grad shape is ", model.lattice_values_monolithic.grad.shape)
                        # wd_coarseness_lattice=np.linspace(0.0, 1e-3, num=model.nr_resolutions)
                        # wd_coarseness_lattice=torch.as_tensor(wd_coarseness_lattice)
                        # wd_coarseness_lattice=wd_coarseness_lattice.view(nr_resolutions, 1, 1)
                        # model.lattice_values_monolithic.grad += model.lattice_values_monolithic*wd_coarseness_lattice

                        #attempt 2 applyg weight decay like done in  https://www.fast.ai/2018/07/02/adam-weight-decay/
                        # wd_coarseness_lattice=np.linspace(0.0, 3, num=model.nr_resolutions)
                        # wd_coarseness_lattice=torch.as_tensor(wd_coarseness_lattice)
                        # wd_coarseness_lattice=wd_coarseness_lattice.view(nr_resolutions, 1, 1)
                        # model.lattice_values_monolithic.data+= (-wd_coarseness_lattice * train_params.lr()) * model.lattice_values_monolithic.data


                        cb.after_backward_pass()

                        #when going from sphere initialization to normal optimization the loss tends to explode a bit
                        #there is a tendency for the curvature to spike once in a while and this seems to solve it
                        grad_clip=40
                        torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=grad_clip, norm_type=2.0)
                        torch.nn.utils.clip_grad_norm(parameters=model_rgb.parameters(), max_norm=grad_clip, norm_type=2.0)
                        torch.nn.utils.clip_grad_norm(parameters=model_bg.parameters(), max_norm=grad_clip, norm_type=2.0)
                        torch.nn.utils.clip_grad_norm(parameters=model_colorcal.parameters(), max_norm=grad_clip, norm_type=2.0)

                        optimizer.step()
                        # scaler.step(optimizer)
                        # scaler.update()

                        if not in_process_of_sphere_init:
                            scheduler_lr_decay.step()

                        if phase.iter_nr==iter_finish_training+1:
                            print("Finished training at iter ", phase.iter_nr)
                            is_in_training_loop=False
                            break 

                        # print("finishing here to debug grad")
                        # exit(1)




                with torch.set_grad_enabled(False):
                    model.eval()
                    model_rgb.eval()
                    model_bg.eval()


                    #save checkpoint
                    if (phase.iter_nr%5000==0 or phase.iter_nr==1) and save_checkpoint:
                    # if (phase.iter_nr%1000==0 or phase.iter_nr==1) and save_checkpoint:
                        # root_folder=os.path.join( os.path.dirname(os.path.abspath(__file__))  , "../") #points at the root of hair_recon package
                        root_folder=checkpoint_path
                        print("saving checkpoint at ", checkpoint_path)
                        print("experiment name is",experiment_name)
                        model.save(root_folder, experiment_name, phase.iter_nr)
                        model_rgb.save(root_folder, experiment_name, phase.iter_nr)
                        model_bg.save(root_folder, experiment_name, phase.iter_nr)
                        model_colorcal.save(root_folder, experiment_name, phase.iter_nr)


                    ###visualize every once in a while
                    should_visualize_things=False
                    if with_viewer:
                        if ngp_gui.m_control_view:
                            should_visualize_things=True
                    # if (phase.iter_nr%500==0 or phase.iter_nr==1 or should_visualize_things or just_finished_sphere_fit) and not in_process_of_sphere_init:
                    if (phase.iter_nr%500==0 or phase.iter_nr==1 or should_visualize_things or just_finished_sphere_fit):

                        if in_process_of_sphere_init:
                            cos_anneal_ratio=1.0
                            forced_variance=1.0

                        print("phase.iter_nr",  phase.iter_nr, "loss ", loss.item() )

                        print("model colorcal weight min max",model_colorcal.weight_delta.min(), model_colorcal.weight_delta.max())
                        print("model colorcal bias min max",model_colorcal.bias.min(), model_colorcal.bias.max())
                        print("model_rgb.lattice_values_monolithic min max ", model_rgb.lattice_values_monolithic.min(), model_rgb.lattice_values_monolithic.max())
                        print("nr_rays_to_create",nr_rays_to_create)

                        #if we have a viewer we visualize there
                        use_only_dense_grid=False
                        if with_viewer:

                            use_only_dense_grid=ngp_gui.m_use_only_dense_grid
                            # Gui.show(tensor2mat(gt_rgb).rgb2bgr(), "gt_rgb")

                            vis_width=150
                            vis_height=150
                            chunk_size=1000
                            if first_time_getting_control or ngp_gui.m_control_view:
                                first_time_getting_control=False
                                frame_controlable=Frame()
                                frame_controlable.from_camera(view.m_camera, vis_width, vis_height)
                                frustum_mesh_controlable=frame_controlable.create_frustum_mesh(0.1)
                                Scene.show(frustum_mesh_controlable,"frustum_mesh_controlable")

                            # #forward all the pixels
                            pred_rgb_img, pred_rgb_fused_img, pred_depth_img, pred_weights_sum_img, ray_end_volumetric, pred_curvature_img,pred_rgb_bg_img, pred_normals_vol_fused_img=run_sdf_and_color_pred(model, model_rgb, model_bg, lattice, lattice_bg, iter_nr_for_anneal, use_only_dense_grid, cos_anneal_ratio, forced_variance, frame_controlable, aabb, occupancy_grid, min_dist_between_samples, max_nr_samples_per_ray, nr_samples_bg, return_curvature=True, chunk_size=500, without_mask=args.without_mask)
                            pred_curvature_img_vis=map_range_tensor(pred_curvature_img, pred_curvature_img.min(), pred_curvature_img.max(), 0.0, 1.0)
                            #view the high curvature parets which will be downweighted by tje robust loss
                            # pred_curvature_img_high_sdf= (pred_curvature_img>0.1).float()
                            # pred_curvature_img_high_sdf_vis=map_range_tensor(pred_curvature_img_high_sdf, pred_curvature_img_high_sdf.min(), pred_curvature_img_high_sdf.max(), 0.0, 1.0)
                            Gui.show(tensor2mat(pred_rgb_img).rgb2bgr(), "pred_rgb_img_control")
                            Gui.show(tensor2mat(pred_rgb_fused_img).rgb2bgr(), "pred_rgb_fused_img")
                            Gui.show(tensor2mat((pred_normals_vol_fused_img+1)*0.5).rgb2bgr(), "pred_normals_vol_fused_img")
                            if pred_rgb_bg_img is not None:
                                Gui.show(tensor2mat(pred_rgb_bg_img).rgb2bgr(), "pred_rgb_bg_img_control")
                            # Gui.show(tensor2mat(pred_rgb_view_dep).rgb2bgr(), "pred_rgb_view_dep")
                            # Gui.show(tensor2mat(torch.sigmoid(pred_rgb_view_dep)).rgb2bgr(), "pred_rgb_view_dep_sig")
                            # Gui.show(tensor2mat(torch.abs(pred_rgb_view_dep)).rgb2bgr(), "pred_rgb_view_dep_abs")
                            Gui.show(tensor2mat(pred_depth_img), "pred_depth_img_control")
                            Gui.show(tensor2mat(pred_weights_sum_img), "pred_weights_sum_img")
                            Gui.show(tensor2mat(pred_curvature_img_vis), "pred_curvature_img_vis")
                            # Gui.show(tensor2mat(pred_curvature_img_high_sdf_vis), "pred_curvature_img_high_sdf_vis")
                            show_points(ray_end_volumetric, "ray_end_volumetric")




                            


                            #sphere trace
                            TIME_START("sphere trace")
                            ray_origins, ray_dirs=model.create_rays(frame_controlable, rand_indices=None) # ray origins and dirs as nr_pixels x 3
                            ray_points_entry, ray_t_entry, ray_points_exit, ray_t_exit, ray_box_intersection=aabb.ray_intersection(ray_origins, ray_dirs)
                            ray_end, ray_end_sdf, ray_end_gradient, ray_t=sphere_trace(25, ray_points_entry, ray_origins, ray_dirs, model, lattice, iter_nr_for_anneal, 0.9, return_gradients=True, use_only_dense_grid=use_only_dense_grid, sdf_clamp=0.2)
                            ray_end_converged, ray_end_gradient_converged=filter_unconverged_points(ray_end, ray_end_sdf, ray_end_gradient) #leaves only the points that are converged
                            ray_end_normal=F.normalize(ray_end_gradient, dim=1)
                            ray_end_normal_vis=(ray_end_normal+1.0)*0.5
                            show_points(ray_end, "ray_end", color_per_vert=ray_end_normal_vis, normal_per_vert=ray_end_normal)
                            ray_end_normal=F.normalize(ray_end_gradient_converged, dim=1)
                            ray_end_normal_vis=(ray_end_normal+1.0)*0.5
                            ray_end_normal_tex=ray_end_normal_vis.view(vis_height, vis_width, 3)
                            ray_end_normal_img=tex2img(ray_end_normal_tex)
                            Gui.show(tensor2mat(ray_end_normal_img), "ray_end_normal_img")
                            TIME_END("sphere trace")


                            #show a certain layer
                            points_layer, sdf, sdf_gradients= sample_sdf_in_layer(model, lattice, iter_nr_for_anneal, use_only_dense_grid, layer_size=100, layer_y_coord=0)
                            sdf_color=colormap(sdf+0.5, "seismic") #we add 0.5 so as to put the zero of the sdf at the center of the colormap
                            show_points(points_layer, "points_layer", color_per_vert=sdf_color)
                        
                    if (phase.iter_nr%5000==0 or phase.iter_nr==1 or just_finished_sphere_fit) and with_tensorboard and not in_process_of_sphere_init:

                        if isinstance(loader_train, DataLoaderPhenorobCP1):
                            frame=random.choice(frames_train)
                        else:
                            frame=phase.loader.get_random_frame() #we just get this frame so that the tensorboard can render from this frame


                        #make from the gt frame a smaller frame until we reach a certain size
                        frame_subsampled=frame.subsample(2.0, subsample_imgs=False)
                        while min(frame_subsampled.width, frame_subsampled.height) >400:
                            frame_subsampled=frame_subsampled.subsample(2.0, subsample_imgs=False)
                        vis_width=frame_subsampled.width
                        vis_height=frame_subsampled.height


                        TIME_START("sphere trace")
                        ray_origins, ray_dirs=model.create_rays(frame_subsampled, rand_indices=None) # ray origins and dirs as nr_pixels x 3
                        ray_points_entry, ray_t_entry, ray_points_exit, ray_t_exit, ray_box_intersection=aabb.ray_intersection(ray_origins, ray_dirs)
                        ray_end, ray_end_sdf, ray_end_gradient, ray_t=sphere_trace(25, ray_points_entry, ray_origins, ray_dirs, model, lattice, iter_nr_for_anneal, 0.9, return_gradients=True, use_only_dense_grid=False, sdf_clamp=0.2)
                        ray_end_converged, ray_end_gradient_converged=filter_unconverged_points(ray_end, ray_end_sdf, ray_end_gradient) #leaves only the points that are converged
                        ray_end_normal=F.normalize(ray_end_gradient_converged, dim=1)
                        ray_end_normal_vis=(ray_end_normal+1.0)*0.5
                        ray_end_normal_tex=ray_end_normal_vis.view(vis_height, vis_width, 3)
                        ray_end_normal_img=tex2img(ray_end_normal_tex)
                        TIME_END("sphere trace")

                        #render all pixels
                        pred_rgb_img, pred_rgb_fused_img, pred_depth_img, pred_weights_sum_img, ray_end_volumetric, pred_curvature_img, pred_rgb_bg_img, pred_normals_vol_fused_img=run_sdf_and_color_pred(model, model_rgb, model_bg, lattice, lattice_bg, iter_nr_for_anneal, False, cos_anneal_ratio, forced_variance, frame_subsampled, aabb, occupancy_grid, min_dist_between_samples, max_nr_samples_per_ray, nr_samples_bg, return_curvature=False, chunk_size=1000, without_mask=args.without_mask)

                        # pred_curvature_img_vis=map_range_tensor(pred_curvature_img, pred_curvature_img.min(), pred_curvature_img.max(), 0.0, 1.0)
                        # print("pred_curvature_img.max()", pred_curvature_img.max())
                        # print("pred_curvature_img.min()", pred_curvature_img.min())
                        # print("pred_curvature_img.mean()", pred_curvature_img.mean())
                        # print("pred_curvature_img.median()", torch.median(pred_curvature_img) )

                        # print("adaptive_loss.alpha",adaptive_loss.alpha())
                        # print("adaptive_loss.scale",adaptive_loss.scale())

                        # print("global_weight_curvature", global_weight_curvature)

                        #show gt, pred_rgb and normals
                        gt_rgb=mat2tensor(frame.rgb_32f, True).to("cuda")
                        cb["tensorboard_callback"].tensorboard_writer.add_image('instant_ngp_2/' + phase.name + '/gt', gt_rgb.squeeze(), phase.iter_nr)
                        cb["tensorboard_callback"].tensorboard_writer.add_image('instant_ngp_2/' + phase.name + '/ray_end_normal_img', ray_end_normal_img.squeeze(), phase.iter_nr)
                        cb["tensorboard_callback"].tensorboard_writer.add_image('instant_ngp_2/' + phase.name + '/pred_rgb_img', pred_rgb_img.squeeze(), phase.iter_nr)
                        cb["tensorboard_callback"].tensorboard_writer.add_image('instant_ngp_2/' + phase.name + '/pred_rgb_fused_img', pred_rgb_fused_img.squeeze(), phase.iter_nr)
                        if pred_rgb_bg_img is not None:
                            cb["tensorboard_callback"].tensorboard_writer.add_image('instant_ngp_2/' + phase.name + '/pred_rgb_bg_img', pred_rgb_bg_img.squeeze(), phase.iter_nr)
                        if pred_normals_vol_fused_img is not None:
                            cb["tensorboard_callback"].tensorboard_writer.add_image('instant_ngp_2/' + phase.name + '/pred_normals_vol_fused_img', torch.clamp(((pred_normals_vol_fused_img+1)*0.5).squeeze(),0,1), phase.iter_nr)
                        # tensorboard_callback.tensorboard_writer.add_image('instant_ngp_2/' + phase.name + '/pred_rgb_view_dep', pred_rgb_view_dep.squeeze(), phase.iter_nr)
                        cb["tensorboard_callback"].tensorboard_writer.add_image('instant_ngp_2/' + phase.name + '/pred_weights_sum_img', pred_weights_sum_img.squeeze(0), phase.iter_nr)
                        # cb["tensorboard_callback"].tensorboard_writer.add_image('instant_ngp_2/' + phase.name + '/pred_curvature_img_vis', pred_curvature_img_vis.squeeze(0), phase.iter_nr)





                if phase.loader.is_finished():
                #     cb.epoch_ended(phase=phase, model=model, save_checkpoint=train_params.save_checkpoint(), checkpoint_path=train_params.checkpoint_path() ) 
                #     cb.phase_ended(phase=phase) 
                    phase.loader.reset()


                if with_viewer:
                    view.update()
