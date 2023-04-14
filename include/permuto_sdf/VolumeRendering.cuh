#pragma once

#include <stdarg.h>



#include "torch/torch.h"

#include <Eigen/Core>

#include "permuto_sdf/OccupancyGrid.cuh" //include RaySamplesPacked

#include "permuto_sdf/pcg32.h"


class VolumeRendering{
public:
    VolumeRendering();
    ~VolumeRendering();

    
    static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> volume_render_nerf(const RaySamplesPacked& ray_samples_packed, const torch::Tensor& rgb_samples, const torch::Tensor& radiance_samples, const torch::Tensor& ray_t_exit, const bool use_ray_t_exit);

    static torch::Tensor compute_dt(const RaySamplesPacked& ray_samples_packed, const torch::Tensor& ray_t_exit, const bool use_ray_t_exit);
    static std::tuple<torch::Tensor, torch::Tensor> cumprod_alpha2transmittance(const RaySamplesPacked& ray_samples_packed, const torch::Tensor& alpha_samples);
    static torch::Tensor integrate_with_weights(const RaySamplesPacked& ray_samples_packed, const torch::Tensor& rgb_samples, const torch::Tensor& weights_samples);
    static torch::Tensor sdf2alpha(const RaySamplesPacked& ray_samples_packed, const torch::Tensor& sdf_samples, const float inv_s, const bool dynamic_inv_s, const float inv_s_multiplier);
    static std::tuple<torch::Tensor, torch::Tensor> sum_over_each_ray(const RaySamplesPacked& ray_samples_packed, const torch::Tensor& sample_values); //sums some tensor that contains some values per sample into a quantity that is per_ray
    static torch::Tensor cumsum_over_each_ray(const RaySamplesPacked& ray_samples_packed, const torch::Tensor& sample_values, const bool inverse); //cumsums some tensor that contains some values per sample into a quantity that is per_ray. If its ivnerse we sum in inverse order so that the first element of the ray has the sum of all the other elements plus itself. The default is that the end of the ray has the sum of all the other elements plus itself
    static torch::Tensor compute_cdf(const RaySamplesPacked& ray_samples_packed, const torch::Tensor& sample_weights);
    static RaySamplesPacked importance_sample(const torch::Tensor& ray_origins, const torch::Tensor& ray_dirs, const RaySamplesPacked& ray_samples_packed, const torch::Tensor& sample_cdf, const int nr_importance_samples, const bool jitter_samples);
    static RaySamplesPacked combine_uniform_samples_with_imp(const torch::Tensor& ray_origins, const torch::Tensor& ray_dirs, const torch::Tensor& ray_t_exit, const RaySamplesPacked& ray_samples_packed, const RaySamplesPacked& ray_samples_imp);

    //backward passes
    static std::tuple<torch::Tensor, torch::Tensor> volume_render_nerf_backward(const torch::Tensor& grad_pred_rgb, const torch::Tensor& grad_bg_transmittance, const torch::Tensor& grad_weight_per_sample, const torch::Tensor& pred_rgb, const RaySamplesPacked& ray_samples_packed, const torch::Tensor& rgb_samples, const torch::Tensor& radiance_samples, const torch::Tensor& ray_t_exit, const bool use_ray_t_exit, const torch::Tensor& bg_transmittance);
    static torch::Tensor cumprod_alpha2transmittance_backward(const torch::Tensor& grad_transmittance, const torch::Tensor& grad_bg_transmittance, const RaySamplesPacked& ray_samples_packed, const torch::Tensor& alpha, const torch::Tensor& transmittance, const torch::Tensor& bg_transmittance, const torch::Tensor& cumsumLV);
    static std::tuple<torch::Tensor, torch::Tensor> integrate_with_weights_backward(const torch::Tensor& grad_pred_rgb, const RaySamplesPacked& ray_samples_packed, const torch::Tensor& rgb_samples, const torch::Tensor& weights_samples, const torch::Tensor& pred_rgb);
    static torch::Tensor sum_over_each_ray_backward(const torch::Tensor& grad_values_sum_per_ray, const torch::Tensor& grad_values_sum_per_sample, const RaySamplesPacked& ray_samples_packed, const torch::Tensor& sample_values);

    static pcg32 m_rng;
    
  

private:
    

  
};
