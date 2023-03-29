import torch
from torch.nn import functional as F
from typing import Optional
import math
import numpy as np

from permuto_sdf  import RaySampler
# from permuto_sdf_py.utils.utils import *

# from hair_recon_py.hair_recon.utils import *
# import hair_recon_py.hair_recon.quaternion as quaternion


HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-6      # float32 only has 7 decimal digits precision

def compute_query_points_from_rays(
		ray_origins: torch.Tensor,
		ray_directions: torch.Tensor,
		near_thresh: float,
		far_thresh: float,
		num_samples: int,
		randomize: Optional[bool] = True,
	) -> (torch.Tensor, torch.Tensor):
		r"""Compute query 3D points given the "bundle" of rays. The near_thresh and far_thresh
		variables indicate the bounds within which 3D points are to be sampled.
		Args:
				ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
					`get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
				ray_directions (torch.Tensor): Direction of each ray in the "bundle" as returned by the
					`get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
				near_thresh (float): The 'near' extent of the bounding volume (i.e., the nearest depth
					coordinate that is of interest/relevance).
				far_thresh (float): The 'far' extent of the bounding volume (i.e., the farthest depth
					coordinate that is of interest/relevance).
				num_samples (int): Number of samples to be drawn along each ray. Samples are drawn
					randomly, whilst trying to ensure "some form of" uniform spacing among them.
				randomize (optional, bool): Whether or not to randomize the sampling of query points.
					By default, this is set to `True`. If disabled (by setting to `False`), we sample
					uniformly spaced points along each ray in the "bundle".
		Returns:
				query_points (torch.Tensor): Query points along each ray
					(shape: :math:`(width, height, num_samples, 3)`).
				depth_values (torch.Tensor): Sampled depth values along each ray
					(shape: :math:`(num_samples)`).
		"""
		# TESTED
		# shape: (num_samples)
		depth_values = torch.linspace(near_thresh, far_thresh, num_samples).to(ray_origins)
		if randomize is True:
				# ray_origins: (width, height, 3)
				# noise_shape = (width, height, num_samples)
				noise_shape = list(ray_directions.shape[:-1]) + [num_samples]
				# depth_values: (num_samples)
				depth_values = (
						depth_values
						+ torch.rand(noise_shape).to(ray_origins)
						* (far_thresh - near_thresh)
						/ num_samples
				)
		# (width, height, num_samples, 3) = (width, height, 1, 3) + (width, height, 1, 3) * (num_samples, 1)
		# query_points:  (width, height, num_samples, 3)
		query_points = (
				ray_origins[..., None, :]
				+ ray_directions[..., None, :] * depth_values[..., :, None]
		)
		# TODO: Double-check that `depth_values` returned is of shape `(num_samples)`.
		return query_points, depth_values

def render_volume_density(
		radiance_field: torch.Tensor, ray_origins: torch.Tensor, depth_values: torch.Tensor
	) -> (torch.Tensor, torch.Tensor, torch.Tensor):
		r"""Differentiably renders a radiance field, given the origin of each ray in the
		"bundle", and the sampled depth values along them.
		Args:
		radiance_field (torch.Tensor): A "field" where, at each query location (X, Y, Z),
			we have an emitted (RGB) color and a volume density (denoted :math:`\sigma` in
			the paper) (shape: :math:`(width, height, num_samples, 4)`).
		ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
			`get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
		depth_values (torch.Tensor): Sampled depth values along each ray
			(shape: :math:`(num_samples)`).
		Returns:
		rgb_map (torch.Tensor): Rendered RGB image (shape: :math:`(width, height, 3)`).
		depth_map (torch.Tensor): Rendered depth image (shape: :math:`(width, height)`).
		acc_map (torch.Tensor): # TODO: Double-check (I think this is the accumulated
			transmittance map).
		"""
		# TESTED
		sigma_a = torch.nn.functional.relu(radiance_field[..., 3])
		# rgb = torch.sigmoid(radiance_field[..., :3])
		rgb = radiance_field[..., :3]
		# print("rgb is ", rgb.min(), rgb.max() )
		one_e_10 = torch.tensor([1e10], dtype=ray_origins.dtype, device=ray_origins.device)
		dists = torch.cat(
				(
						depth_values[..., 1:] - depth_values[..., :-1],
						one_e_10.expand(depth_values[..., :1].shape),
				),
				dim=-1,
		)
		alpha = 1.0 - torch.exp(-sigma_a * dists)
		weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

		rgb_map = (weights[..., None] * rgb).sum(dim=-2)
		depth_map = (weights * depth_values).sum(dim=-1)
		acc_map = weights.sum(-1)

		return rgb_map, depth_map, acc_map


def volume_render_radiance_field(
		radiance_field,
		depth_values,
		ray_directions,
		radiance_field_noise_std=0.0,
		white_background=False,
	):
		# TESTED
		one_e_10 = torch.tensor(
				[1e10], dtype=ray_directions.dtype, device=ray_directions.device
		)
		dists = torch.cat(
				(
						depth_values[..., 1:] - depth_values[..., :-1],
						one_e_10.expand(depth_values[..., :1].shape),
				),
				dim=-1,
		)
		dists = dists * ray_directions[..., None, :].norm(p=2, dim=-1)

		# rgb = torch.sigmoid(radiance_field[..., :3])
		rgb = radiance_field[..., :3]
		noise = 0.0
		if radiance_field_noise_std > 0.0:
				noise = (
						torch.randn(
								radiance_field[..., 3].shape,
								dtype=radiance_field.dtype,
								device=radiance_field.device,
						)
						* radiance_field_noise_std
				)
				# noise = noise.to(radiance_field)
		sigma_a = torch.nn.functional.relu(radiance_field[..., 3] + noise)
		# print("sigma_a", sigma_a.shape)
		# print("dists", dists.shape)
		alpha = 1.0 - torch.exp(-sigma_a * dists)
		weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

		rgb_map = weights[..., None] * rgb
		rgb_map = rgb_map.sum(dim=-2)
		depth_map = weights * depth_values
		depth_map = depth_map.sum(dim=-1)
		# depth_map = (weights * depth_values).sum(dim=-1)
		acc_map = weights.sum(dim=-1)
		disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

		if white_background:
				rgb_map = rgb_map + (1.0 - acc_map[..., None])

		return rgb_map, disp_map, acc_map, weights, depth_map


def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
		r"""Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.
		Args:
		tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
			is to be computed.
		Returns:
		cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
			tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
		"""
		# TESTED
		# Only works for the last dimension (dim=-1)
		dim = -1
		# Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
		cumprod = torch.cumprod(tensor, dim)
		# "Roll" the elements along dimension 'dim' by 1 element.
		cumprod = torch.roll(cumprod, 1, dim)
		# Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
		cumprod[..., 0] = 1.0

		return cumprod


def gather_cdf_util(cdf, inds):
		r"""A very contrived way of mimicking a version of the tf.gather()
		call used in the original impl.
		"""
		orig_inds_shape = inds.shape
		inds_flat = [inds[i].view(-1) for i in range(inds.shape[0])]
		valid_mask = [
				torch.where(ind >= cdf.shape[1], torch.zeros_like(ind), torch.ones_like(ind))
				for ind in inds_flat
		]
		inds_flat = [
				torch.where(ind >= cdf.shape[1], (cdf.shape[1] - 1) * torch.ones_like(ind), ind)
				for ind in inds_flat
		]
		cdf_flat = [cdf[i][ind] for i, ind in enumerate(inds_flat)]
		cdf_flat = [cdf_flat[i] * valid_mask[i] for i in range(len(cdf_flat))]
		cdf_flat = [
				cdf_chunk.reshape([1] + list(orig_inds_shape[1:])) for cdf_chunk in cdf_flat
		]
		return torch.cat(cdf_flat, dim=0)


def sample_pdf(bins, weights, num_samples, det=False):
		# TESTED (Carefully, line-to-line).
		# But chances of bugs persist; haven't integration-tested with
		# training routines.

		# Get pdf
		weights = weights + 1e-5  # prevent nans
		pdf = weights / weights.sum(-1).unsqueeze(-1)
		cdf = torch.cumsum(pdf, -1)
		cdf = torch.cat((torch.zeros_like(cdf[..., :1]), cdf), -1)

		# Take uniform samples
		if det:
				u = torch.linspace(0.0, 1.0, num_samples).to(weights)
				u = u.expand(list(cdf.shape[:-1]) + [num_samples])
		else:
				u = torch.rand(list(cdf.shape[:-1]) + [num_samples]).to(weights)

		# Invert CDF
		# inds = torch.searchsorted(
				# cdf.contiguous(), u.contiguous(), right=False
		# )
		inds = torchsearchsorted.searchsorted(
				cdf.contiguous(), u.contiguous(), side="right"
		)
		below = torch.max(torch.zeros_like(inds), inds - 1)
		above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
		inds_g = torch.stack((below, above), -1)
		orig_inds_shape = inds_g.shape

		cdf_g = gather_cdf_util(cdf, inds_g)
		bins_g = gather_cdf_util(bins, inds_g)

		denom = cdf_g[..., 1] - cdf_g[..., 0]
		denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
		t = (u - cdf_g[..., 0]) / denom
		samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

		return samples


#from https://github.com/Kai-46/nerfplusplus/blob/master/ddp_train_nerf.py
def sample_pdf2(bins, weights, N_samples, det=False):
		'''
		:param bins: tensor of shape [..., M+1], M is the number of bins
		:param weights: tensor of shape [..., M]
		:param N_samples: number of samples along each ray
		:param det: if True, will perform deterministic sampling
		:return: [..., N_samples]
		'''
		# Get pdf
		weights = weights + TINY_NUMBER      # prevent nans
		pdf = weights / torch.sum(weights, dim=-1, keepdim=True)    # [..., M]
		cdf = torch.cumsum(pdf, dim=-1)                             # [..., M]
		cdf = torch.cat([torch.zeros_like(cdf[..., 0:1]), cdf], dim=-1)     # [..., M+1]

		# Take uniform samples
		dots_sh = list(weights.shape[:-1])
		M = weights.shape[-1]

		min_cdf = 0.00
		max_cdf = 1.00       # prevent outlier samples

		if det:
				u = torch.linspace(min_cdf, max_cdf, N_samples, device=bins.device)
				u = u.view([1]*len(dots_sh) + [N_samples]).expand(dots_sh + [N_samples,])   # [..., N_samples]
		else:
				sh = dots_sh + [N_samples]
				u = torch.rand(*sh, device=bins.device) * (max_cdf - min_cdf) + min_cdf        # [..., N_samples]

		# Invert CDF
		# [..., N_samples, 1] >= [..., 1, M] ----> [..., N_samples, M] ----> [..., N_samples,]
		above_inds = torch.sum(u.unsqueeze(-1) >= cdf[..., :M].unsqueeze(-2), dim=-1).long()

		# random sample inside each bin
		below_inds = torch.clamp(above_inds-1, min=0)
		inds_g = torch.stack((below_inds, above_inds), dim=-1)     # [..., N_samples, 2]

		cdf = cdf.unsqueeze(-2).expand(dots_sh + [N_samples, M+1])   # [..., N_samples, M+1]
		cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)       # [..., N_samples, 2]

		bins = bins.unsqueeze(-2).expand(dots_sh + [N_samples, M+1])    # [..., N_samples, M+1]
		bins_g = torch.gather(input=bins, dim=-1, index=inds_g)  # [..., N_samples, 2]

		# fix numeric issue
		denom = cdf_g[..., 1] - cdf_g[..., 0]      # [..., N_samples]
		denom = torch.where(denom<TINY_NUMBER, torch.ones_like(denom), denom)
		t = (u - cdf_g[..., 0]) / denom

		samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0] + TINY_NUMBER)

		return samples

#https://github.com/Totoro97/NeuS/blob/2708e43ed71bcd18dc26b2a1a9a92ac15884111c/models/renderer.py#L131
def neus_sample_pdf(bins, weights, n_samples, deterministic=False):
		# This implementation is from NeRF
		# Get pdf
		weights = weights + 1e-5  # prevent nans
		pdf = weights / torch.sum(weights, -1, keepdim=True)
		cdf = torch.cumsum(pdf, -1)
		cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
		# Take uniform samples
		if deterministic:
				u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
				u = u.expand(list(cdf.shape[:-1]) + [n_samples])
		else:
				u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

		# Invert CDF
		u = u.contiguous()
		inds = torch.searchsorted(cdf, u, right=True)
		below = torch.max(torch.zeros_like(inds - 1), inds - 1)
		above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
		inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

		matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
		cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
		bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

		denom = (cdf_g[..., 1] - cdf_g[..., 0])
		denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
		t = (u - cdf_g[..., 0]) / denom
		samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

		return samples


def importance_sample(z_vals, weights, nr_samples_per_ray_fine, perturb):
		z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
		# z_samples = sample_pdf(
		z_samples = sample_pdf2(
				z_vals_mid,
				weights[..., 1:-1],
				nr_samples_per_ray_fine,
				det=(perturb == 0.0),
		)
		z_samples = z_samples.detach()
		z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)

		return z_vals



#frm nerfies utils
def log1p_safe(x):
	"""The same as tf.math.log1p(x), but clamps the input to prevent NaNs."""
	return torch.log1p(torch.minimum(x, torch.tensor(3e37) ))

def expm1_safe(x):
	"""The same as tf.math.expm1(x), but clamps the input to prevent NaNs."""
	return torch.expm1(torch.minimum(x, torch.tensor(87.5)  ))


#from, nerfies https://github.com/google/nerfies/blob/d0940fb16b3473ce49d192ebb0b6589d69ce2dee/nerfies/utils.py#L265
def general_loss_with_squared_residual(squared_x, alpha, scale):
	r"""The general loss that takes a squared residual.
	This fuses the sqrt operation done to compute many residuals while preserving
	the square in the loss formulation.
	This implements the rho(x, \alpha, c) function described in "A General and
	Adaptive Robust Loss Function", Jonathan T. Barron,
	https://arxiv.org/abs/1701.03077.
	Args:
		squared_x: The residual for which the loss is being computed. x can have
			any shape, and alpha and scale will be broadcasted to match x's shape if
			necessary.
		alpha: The shape parameter of the loss (\alpha in the paper), where more
			negative values produce a loss with more robust behavior (outliers "cost"
			less), and more positive values produce a loss with less robust behavior
			(outliers are penalized more heavily). Alpha can be any value in
			[-infinity, infinity], but the gradient of the loss with respect to alpha
			is 0 at -infinity, infinity, 0, and 2. Varying alpha allows for smooth
			interpolation between several discrete robust losses:
				alpha=-Infinity: Welsch/Leclerc Loss.
				alpha=-2: Geman-McClure loss.
				alpha=0: Cauchy/Lortentzian loss.
				alpha=1: Charbonnier/pseudo-Huber loss.
				alpha=2: L2 loss.
		scale: The scale parameter of the loss. When |x| < scale, the loss is an
			L2-like quadratic bowl, and when |x| > scale the loss function takes on a
			different shape according to alpha.
	Returns:
		The losses for each element of x, in the same shape as x.
	"""
	eps = torch.finfo(torch.float32).eps
	eps=torch.tensor(eps)

	alpha=torch.tensor(alpha)
	scale=torch.tensor(scale)

	# print("alpha is ", alpha)

	# This will be used repeatedly.
	squared_scaled_x = squared_x / (scale ** 2)

	# The loss when alpha == 2.
	loss_two = 0.5 * squared_scaled_x
	# The loss when alpha == 0.
	loss_zero = log1p_safe(0.5 * squared_scaled_x)
	# The loss when alpha == -infinity.
	loss_neginf = -torch.expm1(-0.5 * squared_scaled_x)
	# The loss when alpha == +infinity.
	loss_posinf = expm1_safe(0.5 * squared_scaled_x)

	# The loss when not in one of the above special cases.
	# Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
	beta_safe = torch.maximum(eps, torch.abs(alpha - 2.)  )
	# Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
	alpha_safe = torch.where(
			torch.greater_equal(alpha, 0.), torch.ones_like(alpha),
			-torch.ones_like(alpha)) * torch.maximum(eps, torch.abs(alpha))
	loss_otherwise = (beta_safe / alpha_safe) * (
			torch.pow(squared_scaled_x / beta_safe + 1., 0.5 * alpha) - 1.)

	# Select which of the cases of the loss to return.
	# loss = torch.where(
			# alpha == -torch.inf, loss_neginf,
			# torch.where(
					# alpha == 0, loss_zero,
					# torch.where(
							# alpha == 2, loss_two,
							# torch.where(alpha == torch.inf, loss_posinf, loss_otherwise))))

	##implemented only the alpha==-2 (Geman-McClure loss) the other ones kinda want a comparison with torch.inf
	if alpha!=-2:
			print("implemented only alpha==-2 for now")
			exit(1)

	loss=loss_otherwise

	return loss

#from a nr_rays x nr_samples tensor of z values, return a new tensor of some z_vals in the middle of each section. Based on Neus paper
def get_midpoint_of_sections(z_vals):
	dists = z_vals[..., 1:] - z_vals[..., :-1]
	z_vals_except_last=z_vals[..., :-1]
	mid_z_vals = z_vals_except_last + dists * 0.5
	#now mid_z_vals is of shape nr_rays x (nr_samples -1)
	#we add another point very close to the last one just we have the same number of samples, so the last section will actually have two samples very close to each other in the middle
	mid_z_vals_last=mid_z_vals[...,-1:]
	mid_z_vals=torch.cat([mid_z_vals, mid_z_vals_last+1e-6],-1)


	# #attempt 2
	# sample_dist=1e-6 #weird, maybe just set this to something very tiny
	# dists = z_vals[..., 1:] - z_vals[..., :-1]
	# dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
	# mid_z_vals = z_vals + dists * 0.5

	return mid_z_vals

def create_rays_from_frame(frame, rand_indices):
	# create grid 
	x_coord= torch.arange(frame.width).view(-1, 1, 1).repeat(1,frame.height, 1)+0.5 #width x height x 1
	y_coord= torch.arange(frame.height).view(1, -1, 1).repeat(frame.width, 1, 1)+0.5 #width x height x 1
	ones=torch.ones(frame.width, frame.height).view(frame.width, frame.height, 1)
	points_2D=torch.cat([x_coord, y_coord, ones],2).transpose(0,1).reshape(-1,3).cuda() #Nx3 we tranpose because we want x cooridnate to be inner most so that we traverse row-wise the image

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
	R=torch.from_numpy( tf_world_cam.linear().copy() ).to("cuda").float()
	t=torch.from_numpy( tf_world_cam.translation().copy() ).to("cuda").view(1,3).float()
	pixels_selected_world_coords=torch.matmul(R, pixels_selected_cam_coords.transpose(0,1).contiguous() ).transpose(0,1).contiguous()  + t
	#get direction
	ray_dirs = pixels_selected_world_coords-t
	ray_dirs=F.normalize(ray_dirs, p=2, dim=1)


	#ray_origins
	ray_origins=t.repeat(nr_rays,1)

	return ray_origins, ray_dirs

def create_samples(args, hyperparams, ray_origins, ray_dirs, jitter_samples, occupancy_grid, bounding_primitive):
	ray_points_entry, ray_t_entry, ray_points_exit, ray_t_exit, does_ray_intersect_box=bounding_primitive.ray_intersection(ray_origins, ray_dirs)

	#foreground samples
	if hyperparams.use_occupancy_grid and occupancy_grid is not None:
		fg_ray_samples_packed=occupancy_grid.compute_samples_in_occupied_regions(ray_origins, ray_dirs, ray_t_entry, ray_t_exit, hyperparams.min_dist_between_samples, hyperparams.max_nr_samples_per_ray, jitter_samples)
		fg_ray_samples_packed=fg_ray_samples_packed.compact_to_valid_samples()

		


	else:
	
		fg_ray_samples_packed= RaySampler.compute_samples_fg(ray_origins, ray_dirs, ray_t_entry, ray_t_exit, hyperparams.min_dist_between_samples, hyperparams.max_nr_samples_per_ray, bounding_primitive.m_radius, bounding_primitive.m_center_tensor, jitter_samples)
		fg_ray_samples_packed=fg_ray_samples_packed.compact_to_valid_samples()


	#create ray samples for bg
	if not args.with_mask:
		bg_ray_samples_packed= RaySampler.compute_samples_bg(ray_origins, ray_dirs, ray_t_exit, hyperparams.nr_samples_bg, bounding_primitive.m_radius, bounding_primitive.m_center_tensor, jitter_samples, False)
	else:
		bg_ray_samples_packed=None


	return fg_ray_samples_packed, bg_ray_samples_packed

