#pragma once


#include "torch/torch.h"

#include "permuto_sdf/pcg32.h"

#include "permuto_sdf/RaySamplesPacked.cuh" //include RaySamplesPacked




class RaySampler{
public:
    RaySampler();
    ~RaySampler();


    

    static RaySamplesPacked compute_samples_bg(const torch::Tensor& ray_origins, const torch::Tensor& ray_dirs, const torch::Tensor& ray_t_exit, const int nr_samples, const float sphere_radius, const torch::Tensor& sphere_center, const bool randomize_position, const bool contract_3d_samples); // contract 3d samples applies a contractions similar to eq10 in mipnerf360 
    static RaySamplesPacked compute_samples_fg(const torch::Tensor& ray_origins, const torch::Tensor& ray_dirs, const torch::Tensor& ray_t_entry, const torch::Tensor& ray_t_exit, const float min_dist_between_samples, const int max_nr_samples_per_ray, const float sphere_radius, const torch::Tensor& sphere_center, const bool randomize_position);

    static pcg32 m_rng;

private:
    

  
};
