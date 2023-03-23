import torch
from typing import Optional
import math
import numpy as np
import torch.nn.functional as F

# from instant_ngp_2_py.utils.aabb import *
from permuto_sdf  import PermutoSDF
from permuto_sdf  import RaySamplesPacked
from permuto_sdf  import VolumeRendering
from skimage import measure
from easypbr  import *



def sdf_loss(surface_sdf, surface_sdf_gradients, offsurface_sdf, offsurface_sdf_gradients, gt_normals, eik_clamp=None): 
	#equation 6 of https://arxiv.org/pdf/2006.09661.pdf

	#all the gradients should be 1 
	all_gradients=torch.cat([surface_sdf_gradients, offsurface_sdf_gradients],0)
	all_sdfs=torch.cat([surface_sdf, offsurface_sdf],0)
	if eik_clamp!=None:
		#for high sdf we have a lower eik loss because we are not so interesting in it being perfect when far away from the surface 
		x=all_sdfs.abs().detach()
		c=eik_clamp #standard_deviation
		eik_weight=torch.exp(- (x*x)/(2*c*c)    ) #We use a guassian, for high sdf we have a lower eik loss because we are not so interesting in it being perfect when far away from the surface 
		eikonal_loss = torch.abs(all_gradients.norm(dim=-1) - 1)*eik_weight
	else:
		eikonal_loss = torch.abs(all_gradients.norm(dim=-1) - 1)

	# eikonal_loss = (all_gradients.norm(dim=-1) - 1)**2

	#points on the surface should have sdf of 0 and they should have the correct normal
	loss_surface_sdf= torch.abs(surface_sdf)
	loss_surface_normal= 1 - F.cosine_similarity(surface_sdf_gradients, gt_normals, dim=-1)
	# print("loss_surface_normal", loss_surface_normal.mean())

	#points off the surface should have sdf that are not very close to zero
	loss_offsurface_high_sdf=torch.exp(-1e2 * torch.abs(offsurface_sdf))



	full_loss= eikonal_loss.mean()*5e1 +  loss_surface_normal.mean()*1e2  + loss_surface_sdf.mean()*3e3  + loss_offsurface_high_sdf.mean()*1e2 #default
	#the following help with getting a smoother sdf that can be better visualizes with isolines
	# full_loss= eikonal_loss.mean()*15e1 +  loss_surface_normal.mean()*1e2  + loss_surface_sdf.mean()*3e3  + loss_offsurface_high_sdf.mean()*1e2
	# full_loss= eikonal_loss.mean()*15e1 +  loss_surface_normal.mean()*1e2  + loss_surface_sdf.mean()*3e3  + loss_offsurface_high_sdf.mean()*5e2
	# full_loss= eikonal_loss.mean()*30e1 +  loss_surface_normal.mean()*1e2  + loss_surface_sdf.mean()*3e3  + loss_offsurface_high_sdf.mean()*5e2
	# full_loss= eikonal_loss.mean()*50e1 +  loss_surface_normal.mean()*1e2  + loss_surface_sdf.mean()*3e3  + loss_offsurface_high_sdf.mean()*5e2
	# full_loss= eikonal_loss.mean()*1e-5 

	# print("full loss", full_loss) 
	# print("eikonal_loss.mean()", eikonal_loss.mean()) 
	# print("loss_surface_normal.mean()", loss_surface_normal.mean()) 
	# print("loss_surface_sdf.mean()", loss_surface_sdf.mean()) 
	# print("loss_offsurface_high_sdf.mean()", loss_offsurface_high_sdf.mean()) 

	return full_loss


def sdf_loss_sphere(offsurface_points, offsurface_sdf, offsurface_sdf_gradients, sphere_radius, sphere_center, distance_scale=1.0):
	# points=torch.cat([surface_points, offsurface_points], 0)
	# sdf=torch.cat([surface_sdf, offsurface_sdf],0)
	# all_gradients=torch.cat([surface_sdf_gradients, offsurface_sdf_gradients],0)
	points=torch.cat([offsurface_points], 0)
	sdf=torch.cat([offsurface_sdf],0)
	all_gradients=torch.cat([offsurface_sdf_gradients],0)

	points_in_sphere_coord=points-torch.as_tensor(sphere_center)
	point_dist_to_center=points_in_sphere_coord.norm(dim=-1, keepdim=True)
	# print("point_dist_to_center", point_dist_to_center)
	dists=(point_dist_to_center  - sphere_radius)*distance_scale
	# print("dists", dists)
	# print("sdf", sdf)

	loss_dists= ((sdf-dists)**2).mean()
	eikonal_loss = (all_gradients.norm(dim=-1) - distance_scale  ) **2
	loss=  loss_dists*3e3 + eikonal_loss.mean()*5e1

	#return also the loss sdf and loss eik
	loss_sdf=loss_dists
	loss_eik=eikonal_loss.mean()

	return loss, loss_sdf, loss_eik

#same as sdf_loss_sphere but takes a list of spheres as input
def sdf_loss_spheres(offsurface_points, offsurface_sdf, offsurface_sdf_gradients, sphere_list, distance_scale=1.0):
	# points=torch.cat([surface_points, offsurface_points], 0)
	# sdf=torch.cat([surface_sdf, offsurface_sdf],0)
	# all_gradients=torch.cat([surface_sdf_gradients, offsurface_sdf_gradients],0)
	points=torch.cat([offsurface_points], 0)
	sdf=torch.cat([offsurface_sdf],0)
	all_gradients=torch.cat([offsurface_sdf_gradients],0)

	for i in range(len(sphere_list)):
		sphere=sphere_list[i]
		sphere_center=sphere.sphere_center
		sphere_radius=sphere.sphere_radius
		points_in_sphere_coord=points-torch.as_tensor(sphere_center)
		point_dist_to_center=points_in_sphere_coord.norm(dim=-1, keepdim=True)
		if i==0:
			dists=(point_dist_to_center  - sphere_radius)*distance_scale
		else: #combine sdfs by min()
			dist_to_cur_sphere=(point_dist_to_center  - sphere_radius)*distance_scale
			dists=torch.min(dists,dist_to_cur_sphere)

	loss_dists= ((sdf-dists)**2).mean()
	# eikonal_loss = torch.abs(all_gradients.norm(dim=-1) - distance_scale)
	eikonal_loss = (all_gradients.norm(dim=-1) - distance_scale  ) **2
	loss=  loss_dists*3e3 + eikonal_loss.mean()*5e1

	#return also the loss sdf and loss eik
	loss_sdf=loss_dists
	loss_eik=eikonal_loss.mean()

	return loss, loss_sdf, loss_eik



#sdf multiplier is preferred to be <1 so that we take more conservative steps and don't overshoot the surface
def sphere_trace(nr_sphere_traces, ray_origins, ray_dirs, model, return_gradients, sdf_multiplier, sdf_converged_tresh, occupancy_grid=None, time_val=None):

	ray_points_entry, ray_t_entry, ray_points_exit, ray_t_exit, is_hit_valid=model.boundary_primitive.ray_intersection(ray_origins, ray_dirs)

	#get the entry point of the ray to the aabb. if an occupancy grid is available, use that one instead
	has_occupancy=occupancy_grid is not None
	if has_occupancy:
		ray_samples_packed=occupancy_grid.compute_first_sample_start_of_occupied_regions(ray_origins, ray_dirs, ray_t_entry, ray_t_exit)
		ray_samples_packed=ray_samples_packed.compact_to_valid_samples() #get only the rays that end up shooting through some occupied region
		pos=ray_samples_packed.samples_pos
		dirs=ray_samples_packed.samples_dirs
		#move position slightyl inside the voxel
		voxel_size=1.0/occupancy_grid.get_nr_voxels_per_dim()
		pos=pos+dirs*voxel_size*0.5
	else:
		#create also ray samples packed and advance them like so when we return from this function we return a ray_samples_packed regardless if we use occupancy or not
		ray_samples_packed=RaySamplesPacked(ray_origins.shape[0], ray_origins.shape[0])
		ray_samples_packed.initialize_with_one_sample_per_ray(ray_points_entry, ray_dirs)
		# pos=ray_points_entry
		# dirs=ray_dirs
		pos=ray_samples_packed.samples_pos
		dirs=ray_samples_packed.samples_dirs
	pts=pos.clone()
	
	ray_converged_flag=torch.zeros_like(pos)[:,0:1].bool() #all rays start as unconverged

	


	for i in range(nr_sphere_traces):

		#get the positions that are converged
		select_cur_iter=torch.logical_not(ray_converged_flag)
		pos_unconverged=pts[ select_cur_iter.repeat(1,3) ].view(-1,3)
		dirs_unconverged=dirs[ select_cur_iter.repeat(1,3) ].view(-1,3)
		if pos_unconverged.shape[0]==0:  #all points are converged
			break;

		#sphere trace
		#if we have a time_val we add it
		if time_val is not None:
			time_tensor=torch.empty((pos_unconverged.shape[0],1))
			time_tensor.fill_(time_val)
			pos_unconverged_time=torch.cat([pos_unconverged,time_tensor],1)
			sdf, feat =model(pos_unconverged_time, model.last_iter_nr)
		else:
			sdf, feat =model(pos_unconverged, model.last_iter_nr)
		pos_unconverged=pos_unconverged+dirs_unconverged*sdf*sdf_multiplier


		#get the if points are now converged
		newly_converged_flag=sdf.abs()<sdf_converged_tresh
		ray_converged_flag[select_cur_iter]=torch.logical_or(ray_converged_flag[select_cur_iter], newly_converged_flag.view(-1) )
		ray_converged_flag=ray_converged_flag.view(-1,1)


		if has_occupancy:
			#check if the new positions are in unnocupied space and if they are move them towards the next occupied voxel
			pos_unconverged, is_within_grid_bounds=occupancy_grid.advance_sample_to_next_occupied_voxel(dirs_unconverged, pos_unconverged)
		else:
			is_within_grid_bounds= model.boundary_primitive.check_point_inside_primitive(pos_unconverged)
		ray_converged_flag[select_cur_iter]=torch.logical_or(ray_converged_flag[select_cur_iter], torch.logical_not(is_within_grid_bounds.view(-1)) )
		ray_converged_flag=ray_converged_flag.view(-1,1)

		#update the new points
		pts[select_cur_iter.repeat(1,3)]=pos_unconverged.view(-1)


	#if we have a time_val we add it
	if time_val is not None:
		time_tensor=torch.empty((pts.shape[0],1))
		time_tensor.fill_(time_val)
		pts_with_potential_time=torch.cat([pts,time_tensor],1)
	else:
		pts_with_potential_time=pts

	 

	#one more tace to get also the normals and the SDF at this end point
	if return_gradients:
		with torch.set_grad_enabled(True):
			# sdf, sdf_gradients, feat =model.get_sdf_and_gradient(points.detach(), model.last_iter_nr)
			sdf, sdf_gradients, geom_feat =model.get_sdf_and_gradient(pts_with_potential_time, model.last_iter_nr)
			sdf_gradients=sdf_gradients.detach()
			sdf_gradients=sdf_gradients[:,0:3]
	else:
		sdf, geom_feat=model(pts_with_potential_time, model.last_iter_nr)
		sdf_gradients=None

	#get also a t value for the ray
	#we assume that all the ray_origins are the same so that we can also deal with pts being from ray samples compacted which means they don't have the same shape as ray_origins
	# t_val=(pts-ray_origins).norm(dim=-1, keepdim=True)
	# t_val=None

	# print("pts end is",pts.shape)

	ray_samples_packed.samples_pos=pts

	return pts, sdf, sdf_gradients, geom_feat, ray_samples_packed

#if the sdf is outside of a threshold, set the ray_end and gradient to zero
def filter_unconverged_points(points, sdf, sdf_gradients, sdf_converged_tresh=0.01):

	sdf_gradients_converged=None

	#remove points which still have an sdf
	is_sdf_converged = (sdf<sdf_converged_tresh)*1.0
	points_converged=points*is_sdf_converged
	if sdf_gradients!=None:
		sdf_gradients_converged=sdf_gradients*is_sdf_converged

	return points_converged, sdf_gradients_converged, is_sdf_converged

def sample_sdf_in_layer(model, lattice, iter_nr, use_only_dense_grid, layer_size, layer_y_coord):
	layer_width=layer_size
	layer_height=layer_size
	x_coord= torch.arange(layer_width).view(-1, 1, 1).repeat(1,layer_height, 1) #width x height x 1
	z_coord= torch.arange(layer_height).view(1, -1, 1).repeat(layer_width, 1, 1) #width x height x 1
	y_coord=torch.zeros(layer_width, layer_height).view(layer_width, layer_height, 1)
	y_coord+=layer_y_coord
	x_coord=x_coord/layer_width-0.5
	z_coord=z_coord/layer_height-0.5
	# x_coord=x_coord*1.5
	# z_coord=z_coord*1.5
	point_layer=torch.cat([x_coord, y_coord, z_coord],2).transpose(0,1).reshape(-1,3).cuda()
	sdf, sdf_gradients, feat, sdf_residual = model.get_sdf_and_gradient(point_layer, lattice, iter_nr, use_only_dense_grid)
	# sdf_color=sdf.abs().repeat(1,3)
	# sdf_color=colormap(sdf+0.5, "seismic") #we add 0.5 so as to put the zero of the sdf at the center of the colormap

	return point_layer, sdf, sdf_gradients

#do it similar to neus because they do it a bit better since they don't allocate all points at the same time
def extract_mesh_from_sdf_model(model, nr_points_per_dim, min_val, max_val, threshold=0):

	torch.set_grad_enabled(False)


	# N = nr_points_per_dim  # grid cells per axis
	# t = np.linspace(-0.5 , 0.5, N, dtype=np.float32)
	# chunk_size=nr_points_per_dim*10 #do not set this to some arbitrary number. There seems to be a bug whn we set it to a fixed value that is not easily divizible by the nr_points per dim. It seems that when we put each sdf chunk into the sdf_full, the bug might occur there. Either way, just leave it to something easily divisible
	N = 64
	X = torch.linspace(min_val, max_val, nr_points_per_dim).split(N)
	Y = torch.linspace(min_val, max_val, nr_points_per_dim).split(N)
	Z = torch.linspace(min_val, max_val, nr_points_per_dim).split(N)

	sdf_full = np.zeros([nr_points_per_dim, nr_points_per_dim, nr_points_per_dim], dtype=np.float32)

	with torch.no_grad():
		for xi, xs in enumerate(X):
			for yi, ys in enumerate(Y):
				for zi, zs in enumerate(Z):
					xx, yy, zz = torch.meshgrid(xs, ys, zs)
					pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)

					sdf_cur, _=model(pts, 9999999) #use a really large iter_nr to ensure that the coarse2fine has finished

					sdf_cur=sdf_cur.reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
					sdf_full[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = sdf_cur


	vertices, faces, normals, values = measure.marching_cubes(sdf_full, threshold )
	print('done', vertices.shape, faces.shape, normals.shape)
	
	vertices = vertices / (nr_points_per_dim - 1.0) * (max_val - min_val) + min_val


	#make mesh
	extracted_mesh=Mesh()
	extracted_mesh.V=vertices
	extracted_mesh.F=faces
	extracted_mesh.NV=-normals

	return extracted_mesh


#do it with the neus networks
def extract_mesh_from_sdf_model_neus(model, nr_points_per_dim, min_val, max_val, threshold=0):

	torch.set_grad_enabled(False)


	# N = nr_points_per_dim  # grid cells per axis
	# t = np.linspace(-0.5 , 0.5, N, dtype=np.float32)
	# chunk_size=nr_points_per_dim*10 #do not set this to some arbitrary number. There seems to be a bug whn we set it to a fixed value that is not easily divizible by the nr_points per dim. It seems that when we put each sdf chunk into the sdf_full, the bug might occur there. Either way, just leave it to something easily divisible
	N = 64
	X = torch.linspace(min_val, max_val, nr_points_per_dim).split(N)
	Y = torch.linspace(min_val, max_val, nr_points_per_dim).split(N)
	Z = torch.linspace(min_val, max_val, nr_points_per_dim).split(N)

	sdf_full = np.zeros([nr_points_per_dim, nr_points_per_dim, nr_points_per_dim], dtype=np.float32)

	with torch.no_grad():
		for xi, xs in enumerate(X):
			for yi, ys in enumerate(Y):
				for zi, zs in enumerate(Z):
					xx, yy, zz = torch.meshgrid(xs, ys, zs)
					pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)

					sdf_cur =model.sdf(pts) #use a really large iter_nr to ensure that the coarse2fine has finished

					sdf_cur=sdf_cur.reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
					sdf_full[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = sdf_cur

	# print("sdf_full min max", sdf_full.min(), sdf_full.max())

	vertices, faces, normals, values = measure.marching_cubes(sdf_full, threshold )
	print('done', vertices.shape, faces.shape, normals.shape)
	
	vertices = vertices / (nr_points_per_dim - 1.0) * (max_val - min_val) + min_val


	#make mesh
	extracted_mesh=Mesh()
	extracted_mesh.V=vertices
	extracted_mesh.F=faces
	extracted_mesh.NV=-normals

	return extracted_mesh

def extract_mesh_from_density_model(model, lattice, nr_points_per_dim, min_val, max_val, threshold=0):

	torch.set_grad_enabled(False)


	# N = nr_points_per_dim  # grid cells per axis
	# t = np.linspace(-0.5 , 0.5, N, dtype=np.float32)
	# chunk_size=nr_points_per_dim*10 #do not set this to some arbitrary number. There seems to be a bug whn we set it to a fixed value that is not easily divizible by the nr_points per dim. It seems that when we put each sdf chunk into the sdf_full, the bug might occur there. Either way, just leave it to something easily divisible
	N = 64
	X = torch.linspace(min_val, max_val, nr_points_per_dim).split(N)
	Y = torch.linspace(min_val, max_val, nr_points_per_dim).split(N)
	Z = torch.linspace(min_val, max_val, nr_points_per_dim).split(N)

	sdf_full = np.zeros([nr_points_per_dim, nr_points_per_dim, nr_points_per_dim], dtype=np.float32)

	with torch.no_grad():
		for xi, xs in enumerate(X):
			for yi, ys in enumerate(Y):
				for zi, zs in enumerate(Z):
					xx, yy, zz = torch.meshgrid(xs, ys, zs)
					pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)

					sdf_cur = model.get_only_density( pts, lattice, 9999999) 

					sdf_cur=sdf_cur.reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
					sdf_full[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = sdf_cur

	print("density_full min max is ", sdf_full.min(), sdf_full.max())

	vertices, faces, normals, values = measure.marching_cubes(sdf_full, threshold )
	print('done', vertices.shape, faces.shape, normals.shape)
	
	vertices = vertices / (nr_points_per_dim - 1.0) * (max_val - min_val) + min_val


	#make mesh
	extracted_mesh=Mesh()
	extracted_mesh.V=vertices
	extracted_mesh.F=faces
	extracted_mesh.NV=-normals

	return extracted_mesh


def importance_sampling_sdf_model(model_sdf, ray_samples_packed, ray_origins, ray_dirs, ray_t_exit, iter_nr_for_anneal):
	#importance sampling for the SDF model
	inv_s_imp_sampling=512
	inv_s_multiplier=1.0
	#
	sdf_sampled_packed, _=model_sdf(ray_samples_packed.samples_pos, iter_nr_for_anneal)
	ray_samples_packed.set_sdf(sdf_sampled_packed) ##set sdf
	alpha=VolumeRendering.sdf2alpha(ray_samples_packed, sdf_sampled_packed, inv_s_imp_sampling, True, inv_s_multiplier)
	alpha=alpha.clip(0.0, 1.0)
	transmittance, bg_transmittance= VolumeRendering.cumprod_alpha2transmittance(ray_samples_packed, 1-alpha + 1e-7)
	weights = alpha * transmittance
	weights_sum, weight_sum_per_sample=VolumeRendering.sum_over_each_ray(ray_samples_packed, weights)
	# weight_sum_per_sample[weight_sum_per_sample==0]=1e-6 #prevent nans
	weight_sum_per_sample=torch.clamp(weight_sum_per_sample, min=1e-6 )
	weights/=weight_sum_per_sample #normalize so that cdf sums up to 1
	cdf=VolumeRendering.compute_cdf(ray_samples_packed, weights)
	# print("cdf min max is ", cdf.min(), cdf.max())
	ray_samples_packed_imp=VolumeRendering.importance_sample(ray_origins, ray_dirs, ray_samples_packed, cdf, 16, model_sdf.training)
	sdf_sampled_packed_imp, _, =model_sdf(ray_samples_packed_imp.samples_pos, iter_nr_for_anneal)
	ray_samples_packed_imp.set_sdf(sdf_sampled_packed_imp) ##set sdf
	ray_samples_combined=VolumeRendering.combine_uniform_samples_with_imp(ray_origins, ray_dirs, ray_t_exit, ray_samples_packed, ray_samples_packed_imp)
	ray_samples_packed=ray_samples_combined#swap
	ray_samples_packed=ray_samples_packed.compact_to_valid_samples() #still need to get the valid ones because we have less samples than allocated
	####SECOND ITER
	inv_s_multiplier=2.0
	sdf_sampled_packed=ray_samples_packed.samples_sdf #we already combined them and have the sdf
	alpha=VolumeRendering.sdf2alpha(ray_samples_packed, sdf_sampled_packed, inv_s_imp_sampling, True, inv_s_multiplier)
	alpha=alpha.clip(0.0, 1.0)
	transmittance, bg_transmittance= VolumeRendering.cumprod_alpha2transmittance(ray_samples_packed, 1-alpha + 1e-7)
	weights = alpha * transmittance
	weights_sum, weight_sum_per_sample=VolumeRendering.sum_over_each_ray(ray_samples_packed, weights)
	weight_sum_per_sample=torch.clamp(weight_sum_per_sample, min=1e-6 )
	weights/=weight_sum_per_sample #normalize so that cdf sums up to 1
	cdf=VolumeRendering.compute_cdf(ray_samples_packed, weights)
	ray_samples_packed_imp=VolumeRendering.importance_sample(ray_origins, ray_dirs, ray_samples_packed, cdf, 16, model_sdf.training)
	ray_samples_packed.remove_sdf() #we fuse this with ray_samples_packed_imp but we don't care about fusing the sdf
	ray_samples_combined=VolumeRendering.combine_uniform_samples_with_imp(ray_origins, ray_dirs, ray_t_exit, ray_samples_packed, ray_samples_packed_imp)
	ray_samples_packed=ray_samples_combined#swap
	ray_samples_packed=ray_samples_packed.compact_to_valid_samples() #still need to get the valid ones because we have less samples than allocated

	return ray_samples_packed





