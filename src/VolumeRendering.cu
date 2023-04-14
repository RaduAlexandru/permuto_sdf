#include "permuto_sdf/VolumeRendering.cuh"

//c++
// #include <string>

#include "UtilsPytorch.h" //contains torch so it has to be added BEFORE any other include because the other ones might include loguru which gets screwed up if torch was included before it

//my stuff
#include "permuto_sdf/VolumeRenderingGPU.cuh"



using torch::Tensor;
using namespace radu::utils;



template <typename T>
T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}


pcg32 VolumeRendering::m_rng;




//CPU code that calls the kernels
VolumeRendering::VolumeRendering()
    {


}

VolumeRendering::~VolumeRendering(){
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> VolumeRendering::volume_render_nerf(const RaySamplesPacked& ray_samples_packed, const torch::Tensor& rgb_samples, const torch::Tensor& radiance_samples, const torch::Tensor& ray_t_exit, const bool use_ray_t_exit){

    CHECK(rgb_samples.dim()==2) << "rgb_samples should have only one dimension correspond to nr_samples x 3. However it has sizes" << rgb_samples.sizes();
    CHECK(radiance_samples.dim()==2) << "radiance_samples should have only one dimension correspond to nr_samples x 1. However it has sizes" << radiance_samples.sizes();
    int nr_rays=ray_samples_packed.ray_start_end_idx.size(0);
    int nr_samples_total=ray_samples_packed.samples_z.size(0);

    //fill the samples
    const dim3 blocks = { (unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1 }; 


    torch::Tensor pred_rgb=torch::zeros({ nr_rays,3 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    torch::Tensor pred_depth=torch::zeros({ nr_rays,1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    torch::Tensor bg_transmittance=torch::zeros({ nr_rays,1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    torch::Tensor weight_per_sample=torch::zeros({ nr_samples_total,1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );


    // TIME_START("volume_render_nerf")
    VolumeRenderingGPU::volume_render_nerf<<<blocks, BLOCK_SIZE>>>(
                nr_rays,
                use_ray_t_exit,
                ray_t_exit.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                rgb_samples.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                radiance_samples.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.max_nr_samples, //useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
                // ray_samples_packed.samples_pos.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                // ray_samples_packed.samples_dirs.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.samples_z.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.samples_dt.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.rays_have_equal_nr_of_samples,
                ray_samples_packed.fixed_nr_of_samples_per_ray,
                //output
                pred_rgb.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                pred_depth.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                bg_transmittance.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                weight_per_sample.packed_accessor32<float,2,torch::RestrictPtrTraits>()
            );
    // TIME_END("volume_render_nerf")


    return std::make_tuple(pred_rgb, pred_depth, bg_transmittance, weight_per_sample);

}


std::tuple<torch::Tensor, torch::Tensor> VolumeRendering::volume_render_nerf_backward(const torch::Tensor& grad_pred_rgb, const torch::Tensor& grad_bg_transmittance, const torch::Tensor& grad_weight_per_sample, const torch::Tensor& pred_rgb, const RaySamplesPacked& ray_samples_packed, const torch::Tensor& rgb_samples, const torch::Tensor& radiance_samples, const torch::Tensor& ray_t_exit, const bool use_ray_t_exit, const torch::Tensor& bg_transmittance){


    CHECK(rgb_samples.dim()==2) << "rgb_samples should have only one dimension correspond to nr_samples x 3. However it has sizes" << rgb_samples.sizes();
    CHECK(radiance_samples.dim()==2) << "radiance_samples should have only one dimension correspond to nr_samples x 1. However it has sizes" << radiance_samples.sizes();
    int nr_rays=ray_samples_packed.ray_start_end_idx.size(0);
    int nr_samples_total=rgb_samples.size(0);

    //fill the samples
    const dim3 blocks = { (unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1 }; 


    //make some tensors for storing the gradients
    torch::Tensor grad_rgb_samples=torch::zeros({ nr_samples_total,3 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    torch::Tensor grad_radiance_samples=torch::zeros({ nr_samples_total,1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );

    

    // TIME_START("volume_render_nerf_backward")
    VolumeRenderingGPU::volume_render_nerf_backward<<<blocks, BLOCK_SIZE>>>(
                nr_rays,
                use_ray_t_exit,
                grad_pred_rgb.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                grad_bg_transmittance.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                grad_weight_per_sample.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                pred_rgb.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_t_exit.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                bg_transmittance.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                rgb_samples.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                radiance_samples.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.max_nr_samples, //useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
                // ray_samples_packed.samples_pos.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                // ray_samples_packed.samples_dirs.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.samples_dt.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.rays_have_equal_nr_of_samples,
                ray_samples_packed.fixed_nr_of_samples_per_ray,
                //output
                grad_rgb_samples.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                grad_radiance_samples.packed_accessor32<float,2,torch::RestrictPtrTraits>()
            );
    // TIME_END("volume_render_nerf_backward")


    return std::make_tuple(grad_rgb_samples, grad_radiance_samples);

}


torch::Tensor VolumeRendering::compute_dt(const RaySamplesPacked& ray_samples_packed, const torch::Tensor& ray_t_exit, const bool use_ray_t_exit){


    int nr_rays=ray_samples_packed.ray_start_end_idx.size(0);
    int nr_samples_total=ray_samples_packed.samples_z.size(0);

    //fill the samples
    const dim3 blocks = { (unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1 }; 


    torch::Tensor dt_samples=torch::zeros({ nr_samples_total,1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );



    VolumeRenderingGPU::compute_dt_gpu<<<blocks, BLOCK_SIZE>>>(
                nr_rays,
                use_ray_t_exit,
                ray_t_exit.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.max_nr_samples, //useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
                ray_samples_packed.samples_z.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.rays_have_equal_nr_of_samples,
                ray_samples_packed.fixed_nr_of_samples_per_ray,
                //output
                dt_samples.packed_accessor32<float,2,torch::RestrictPtrTraits>()
            );

    //attempt 2 to just return the distance between samples which is same for all samples in the ray
    // return ray_samples_packed.samples_dt;

    return dt_samples;

}

std::tuple<torch::Tensor, torch::Tensor> VolumeRendering::cumprod_alpha2transmittance(const RaySamplesPacked& ray_samples_packed, const torch::Tensor& alpha_samples){


    int nr_rays=ray_samples_packed.ray_start_end_idx.size(0);
    int nr_samples_total=ray_samples_packed.samples_z.size(0);

    //fill the samples
    const dim3 blocks = { (unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1 }; 


    torch::Tensor transmittance_samples=torch::zeros({ nr_samples_total,1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    //if we have no samples for this ray, then the bg_transmittance stays as 1 so the background can be fully visible
    torch::Tensor bg_transmittance=torch::ones({ nr_rays,1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  ); 



    VolumeRenderingGPU::cumprod_alpha2transmittance_gpu<<<blocks, BLOCK_SIZE>>>(
                nr_rays,
                ray_samples_packed.max_nr_samples, //useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
                ray_samples_packed.ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.rays_have_equal_nr_of_samples,
                ray_samples_packed.fixed_nr_of_samples_per_ray,
                alpha_samples.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                //output
                transmittance_samples.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                bg_transmittance.packed_accessor32<float,2,torch::RestrictPtrTraits>()
            );


    return std::make_tuple(transmittance_samples, bg_transmittance);


}


torch::Tensor VolumeRendering::integrate_with_weights(const RaySamplesPacked& ray_samples_packed, const torch::Tensor& rgb_samples, const torch::Tensor& weights_samples){

    int nr_rays=ray_samples_packed.ray_start_end_idx.size(0);
    // int nr_samples_total=ray_samples_packed.samples_z.size(0);

    //fill the samples
    const dim3 blocks = { (unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1 }; 


    torch::Tensor pred_rgb_per_ray=torch::zeros({ nr_rays,3 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );



    VolumeRenderingGPU::integrate_with_weights_gpu<<<blocks, BLOCK_SIZE>>>(
                nr_rays,
                ray_samples_packed.max_nr_samples, //useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
                ray_samples_packed.ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.rays_have_equal_nr_of_samples,
                ray_samples_packed.fixed_nr_of_samples_per_ray,
                rgb_samples.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                weights_samples.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                //output
                pred_rgb_per_ray.packed_accessor32<float,2,torch::RestrictPtrTraits>()
            );


    return pred_rgb_per_ray;

}

torch::Tensor VolumeRendering::sdf2alpha(const RaySamplesPacked& ray_samples_packed, const torch::Tensor& sdf_samples, const float inv_s, const bool dynamic_inv_s, const float inv_s_multiplier){


    int nr_rays=ray_samples_packed.ray_start_end_idx.size(0);
    int nr_samples_total=ray_samples_packed.samples_z.size(0);

    //fill the samples
    const dim3 blocks = { (unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1 }; 


    torch::Tensor alpha_samples=torch::zeros({ nr_samples_total,1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );



    VolumeRenderingGPU::sdf2alpha_gpu<<<blocks, BLOCK_SIZE>>>(
                nr_rays,
                ray_samples_packed.max_nr_samples, //useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
                ray_samples_packed.ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.ray_fixed_dt.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.samples_dt.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.rays_have_equal_nr_of_samples,
                ray_samples_packed.fixed_nr_of_samples_per_ray,
                sdf_samples.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                inv_s,
                // inv_s_min,
                // inv_s_max,
                dynamic_inv_s,
                inv_s_multiplier,
                //output
                alpha_samples.packed_accessor32<float,2,torch::RestrictPtrTraits>()
            );


    return alpha_samples;

}


std::tuple<torch::Tensor, torch::Tensor> VolumeRendering::sum_over_each_ray(const RaySamplesPacked& ray_samples_packed, const torch::Tensor& sample_values){

    int nr_rays=ray_samples_packed.ray_start_end_idx.size(0);
    int nr_samples_total=ray_samples_packed.samples_z.size(0);
    CHECK(sample_values.size(0)==nr_samples_total) <<"sample_values should have size of nr_samples_total x 1. but it has " << sample_values.sizes();
    CHECK(sample_values.size(1)<=3 || sample_values.size(1)==32) <<"sample_values should ahve up to 4 values value per sample because I haven't implemented a better code to support multiple values " << sample_values.sizes();
    int val_dim=sample_values.size(1);

    //fill the samples
    const dim3 blocks = { (unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1 }; 


    torch::Tensor values_sum=torch::zeros({ nr_rays,val_dim },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    torch::Tensor values_sum_stored_per_sample=torch::zeros({ nr_samples_total,val_dim },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );


    if(val_dim==1){
        VolumeRenderingGPU::sum_over_each_ray_gpu<1><<<blocks, BLOCK_SIZE>>>(
                nr_rays,
                ray_samples_packed.max_nr_samples, //useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
                ray_samples_packed.ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.rays_have_equal_nr_of_samples,
                ray_samples_packed.fixed_nr_of_samples_per_ray,
                sample_values.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                //output
                values_sum.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                values_sum_stored_per_sample.packed_accessor32<float,2,torch::RestrictPtrTraits>()
            );
    }else if(val_dim==2){
         VolumeRenderingGPU::sum_over_each_ray_gpu<2><<<blocks, BLOCK_SIZE>>>(
                nr_rays,
                ray_samples_packed.max_nr_samples, //useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
                ray_samples_packed.ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.rays_have_equal_nr_of_samples,
                ray_samples_packed.fixed_nr_of_samples_per_ray,
                sample_values.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                //output
                values_sum.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                values_sum_stored_per_sample.packed_accessor32<float,2,torch::RestrictPtrTraits>()
            );
    }else if(val_dim==3){
        VolumeRenderingGPU::sum_over_each_ray_gpu<3><<<blocks, BLOCK_SIZE>>>(
                nr_rays,
                ray_samples_packed.max_nr_samples, //useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
                ray_samples_packed.ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.rays_have_equal_nr_of_samples,
                ray_samples_packed.fixed_nr_of_samples_per_ray,
                sample_values.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                //output
                values_sum.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                values_sum_stored_per_sample.packed_accessor32<float,2,torch::RestrictPtrTraits>()
            );
    }else if(val_dim==32){
        VolumeRenderingGPU::sum_over_each_ray_gpu<32><<<blocks, BLOCK_SIZE>>>(
                nr_rays,
                ray_samples_packed.max_nr_samples, //useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
                ray_samples_packed.ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.rays_have_equal_nr_of_samples,
                ray_samples_packed.fixed_nr_of_samples_per_ray,
                sample_values.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                //output
                values_sum.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                values_sum_stored_per_sample.packed_accessor32<float,2,torch::RestrictPtrTraits>()
            );
    }else{
        LOG(FATAL) <<"Val dim not implemented yet";
    }



    return std::make_tuple(values_sum, values_sum_stored_per_sample);

}

torch::Tensor VolumeRendering::cumsum_over_each_ray(const RaySamplesPacked& ray_samples_packed, const torch::Tensor& sample_values, const bool inverse){

    int nr_rays=ray_samples_packed.ray_start_end_idx.size(0);
    int nr_samples_total=ray_samples_packed.samples_z.size(0);
    CHECK(sample_values.size(0)==nr_samples_total) <<"sample_values should have size of nr_samples_total x 1. but it has " << sample_values.sizes();
    CHECK(sample_values.size(1)==1) <<"sample_values should ahve only 1 value per sample because I haven't implemented a better code to support multiple values " << sample_values.sizes();

    //fill the samples
    const dim3 blocks = { (unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1 }; 


    torch::Tensor values_cumsum_stored_per_sample=torch::zeros({ nr_samples_total,1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );



    VolumeRenderingGPU::cumsum_over_each_ray_gpu<<<blocks, BLOCK_SIZE>>>(
                nr_rays,
                ray_samples_packed.max_nr_samples, //useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
                ray_samples_packed.ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.rays_have_equal_nr_of_samples,
                ray_samples_packed.fixed_nr_of_samples_per_ray,
                sample_values.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                inverse,
                //output
                values_cumsum_stored_per_sample.packed_accessor32<float,2,torch::RestrictPtrTraits>()
            );


    return values_cumsum_stored_per_sample;

}


torch::Tensor VolumeRendering::compute_cdf(const RaySamplesPacked& ray_samples_packed, const torch::Tensor& sample_weights){

    int nr_rays=ray_samples_packed.ray_start_end_idx.size(0);
    int nr_samples_total=ray_samples_packed.samples_z.size(0);
    CHECK(sample_weights.size(0)==nr_samples_total) <<"Weights should have size of nr_samples_total x 1. but it has " << sample_weights.sizes();
    CHECK(sample_weights.size(1)==1) <<"Weights should ahve only 1 value per sample " << sample_weights.sizes();

    //fill the samples
    const dim3 blocks = { (unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1 }; 


    torch::Tensor samples_cdf=torch::zeros({ nr_samples_total,1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );



    VolumeRenderingGPU::compute_cdf_gpu<<<blocks, BLOCK_SIZE>>>(
                nr_rays,
                ray_samples_packed.max_nr_samples, //useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
                ray_samples_packed.ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.rays_have_equal_nr_of_samples,
                ray_samples_packed.fixed_nr_of_samples_per_ray,
                sample_weights.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                //output
                samples_cdf.packed_accessor32<float,2,torch::RestrictPtrTraits>()
            );


    return samples_cdf;

}

RaySamplesPacked VolumeRendering::importance_sample(const torch::Tensor& ray_origins, const torch::Tensor& ray_dirs, const RaySamplesPacked& ray_samples_packed, const torch::Tensor& sample_cdf, const int nr_importance_samples, const bool jitter_samples){

    int nr_rays=ray_samples_packed.ray_start_end_idx.size(0);
    int nr_samples_total=ray_samples_packed.samples_z.size(0);
    CHECK(sample_cdf.size(0)==nr_samples_total) <<"CDF should have size of nr_samples_total x 1. but it has " << sample_cdf.sizes();
    CHECK(sample_cdf.size(1)==1) <<"CDF should ahve only 1 value per sample " << sample_cdf.sizes();

    int nr_samples_imp_maximum=nr_rays*nr_importance_samples; 
    RaySamplesPacked ray_samples_imp(nr_rays, nr_samples_imp_maximum);
    ray_samples_imp.rays_have_equal_nr_of_samples=true;
    ray_samples_imp.fixed_nr_of_samples_per_ray=nr_importance_samples;

    //fill the samples
    const dim3 blocks = { (unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1 }; 



    VolumeRenderingGPU::importance_sample_gpu<<<blocks, BLOCK_SIZE>>>(
                nr_rays,
                ray_origins.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_dirs.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                //input_ray_samples_packed
                ray_samples_packed.max_nr_samples, //useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
                ray_samples_packed.ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.ray_fixed_dt.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.rays_have_equal_nr_of_samples,
                ray_samples_packed.fixed_nr_of_samples_per_ray,
                ray_samples_packed.samples_z.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                sample_cdf.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                nr_importance_samples,
                m_rng,
                jitter_samples,
                //output
                // samples_cdf.packed_accessor32<float,2,torch::RestrictPtrTraits>()
                ray_samples_imp.samples_pos.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_imp.samples_dirs.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_imp.samples_z.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                // ray_samples_imp.samples_dt.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                // ray_samples_imp.ray_fixed_dt.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_imp.ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>()
                // ray_samples_imp.cur_nr_samples.packed_accessor32<int,1,torch::RestrictPtrTraits>()
            );
    
    if(jitter_samples){
        m_rng.advance();
    }


    return ray_samples_imp;


}


RaySamplesPacked VolumeRendering::combine_uniform_samples_with_imp(const torch::Tensor& ray_origins, const torch::Tensor& ray_dirs, const torch::Tensor& ray_t_exit, const RaySamplesPacked& ray_samples_packed, const RaySamplesPacked& ray_samples_imp){

    CHECK(ray_samples_imp.rays_have_equal_nr_of_samples) <<"We are assuming that the importance samples have all an equal nr of samples per ray";
    CHECK(ray_samples_packed.has_sdf==ray_samples_imp.has_sdf) <<"They are supposed to both has or not have sdf";


    int nr_rays=ray_samples_packed.ray_start_end_idx.size(0);
    int nr_samples_uniform_total=ray_samples_packed.samples_z.size(0);
    int nr_samples_imp_total=ray_samples_imp.max_nr_samples;
    CHECK(nr_rays*ray_samples_imp.fixed_nr_of_samples_per_ray == nr_samples_imp_total) <<"This should be equal";

    int nr_samples_combined_maximum=nr_samples_uniform_total+nr_samples_imp_total; 
    RaySamplesPacked ray_samples_combined(nr_rays, nr_samples_combined_maximum);
    ray_samples_combined.has_sdf=ray_samples_packed.has_sdf;

    //fill the samples
    const dim3 blocks = { (unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1 }; 



    VolumeRenderingGPU::combine_uniform_samples_with_imp_gpu<<<blocks, BLOCK_SIZE>>>(
                nr_rays,
                ray_origins.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_dirs.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_t_exit.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                //input_ray_samples_packed
                ray_samples_packed.max_nr_samples, //useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
                ray_samples_packed.ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.ray_fixed_dt.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.rays_have_equal_nr_of_samples,
                ray_samples_packed.fixed_nr_of_samples_per_ray,
                ray_samples_packed.samples_z.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.samples_sdf.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.has_sdf,
                //imp_samples
                ray_samples_imp.max_nr_samples, //useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
                ray_samples_imp.ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                ray_samples_imp.ray_fixed_dt.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_imp.rays_have_equal_nr_of_samples,
                ray_samples_imp.fixed_nr_of_samples_per_ray,
                ray_samples_imp.samples_z.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_imp.samples_sdf.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_imp.has_sdf,
                //combined stuff just for sanity checking
                ray_samples_combined.max_nr_samples,
                //output
                ray_samples_combined.samples_pos.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_combined.samples_dirs.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_combined.samples_z.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_combined.samples_dt.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_combined.samples_sdf.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_combined.ray_fixed_dt.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_combined.ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                ray_samples_combined.cur_nr_samples.packed_accessor32<int,1,torch::RestrictPtrTraits>()
            );
    
    


    return ray_samples_combined;

}




torch::Tensor VolumeRendering::cumprod_alpha2transmittance_backward(const torch::Tensor& grad_transmittance, const torch::Tensor& grad_bg_transmittance, const RaySamplesPacked& ray_samples_packed, const torch::Tensor& alpha, const torch::Tensor& transmittance, const torch::Tensor& bg_transmittance, const torch::Tensor& cumsumLV){


    int nr_rays=ray_samples_packed.ray_start_end_idx.size(0);
    int nr_samples_total=ray_samples_packed.samples_z.size(0);
    CHECK(grad_transmittance.size(0)==nr_samples_total) <<"grad_transmittance should have size of nr_samples_total x 1. but it has " << grad_transmittance.sizes();

    //fill the samples
    const dim3 blocks = { (unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1 }; 


    torch::Tensor grad_alpha_samples=torch::zeros({ nr_samples_total,1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );



    VolumeRenderingGPU::cumprod_alpha2transmittance_backward_gpu<<<blocks, BLOCK_SIZE>>>(
                nr_rays,
                ray_samples_packed.max_nr_samples, //useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
                ray_samples_packed.ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.rays_have_equal_nr_of_samples,
                ray_samples_packed.fixed_nr_of_samples_per_ray,
                grad_transmittance.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                grad_bg_transmittance.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                alpha.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                transmittance.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                bg_transmittance.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                cumsumLV.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                //output
                grad_alpha_samples.packed_accessor32<float,2,torch::RestrictPtrTraits>()
            );


    return grad_alpha_samples;

}


std::tuple<torch::Tensor, torch::Tensor> VolumeRendering::integrate_with_weights_backward(const torch::Tensor& grad_pred_rgb, const RaySamplesPacked& ray_samples_packed, const torch::Tensor& rgb_samples, const torch::Tensor& weights_samples, const torch::Tensor& pred_rgb){

    int nr_rays=ray_samples_packed.ray_start_end_idx.size(0);
    int nr_samples_total=ray_samples_packed.samples_z.size(0);
    CHECK(grad_pred_rgb.size(0)==nr_rays) <<"grad_pred_rgb should have size of nr_samples_total x 3. but it has " << grad_pred_rgb.sizes();
    CHECK(grad_pred_rgb.size(1)==3) <<"grad_pred_rgb should have size of nr_samples_total x 3. but it has " << grad_pred_rgb.sizes();

    //fill the samples
    const dim3 blocks = { (unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1 }; 


    torch::Tensor grad_rgb_samples=torch::zeros({ nr_samples_total,3 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    torch::Tensor grad_weights_samples=torch::zeros({ nr_samples_total,1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );



    VolumeRenderingGPU::integrate_with_weights_backward_gpu<<<blocks, BLOCK_SIZE>>>(
                nr_rays,
                ray_samples_packed.max_nr_samples, //useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
                ray_samples_packed.ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.rays_have_equal_nr_of_samples,
                ray_samples_packed.fixed_nr_of_samples_per_ray,
                grad_pred_rgb.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                rgb_samples.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                weights_samples.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                pred_rgb.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                //output
                grad_rgb_samples.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                grad_weights_samples.packed_accessor32<float,2,torch::RestrictPtrTraits>()
            );


    return std::make_tuple(grad_rgb_samples, grad_weights_samples);

}


torch::Tensor VolumeRendering::sum_over_each_ray_backward(const torch::Tensor& grad_values_sum_per_ray, const torch::Tensor& grad_values_sum_per_sample, const RaySamplesPacked& ray_samples_packed, const torch::Tensor& sample_values){



    int nr_rays=ray_samples_packed.ray_start_end_idx.size(0);
    int nr_samples_total=ray_samples_packed.samples_z.size(0);
    CHECK(grad_values_sum_per_ray.size(0)==nr_rays) <<"grad_values_sum_per_ray should have size of nr_rays x 1. but it has " << grad_values_sum_per_ray.sizes();
    CHECK(grad_values_sum_per_sample.size(0)==nr_samples_total) <<"grad_values_sum_per_sample should have size nr_sample x 1 " << grad_values_sum_per_sample.sizes();
    int val_dim=sample_values.size(1);

    //fill the samples
    const dim3 blocks = { (unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1 }; 


    torch::Tensor grad_sample_values=torch::zeros({ nr_samples_total,val_dim },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );


    if(val_dim==1){
        VolumeRenderingGPU::sum_over_each_ray_backward_gpu<1><<<blocks, BLOCK_SIZE>>>(
                nr_rays,
                ray_samples_packed.max_nr_samples, //useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
                ray_samples_packed.ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.rays_have_equal_nr_of_samples,
                ray_samples_packed.fixed_nr_of_samples_per_ray,
                grad_values_sum_per_ray.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                grad_values_sum_per_sample.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                sample_values.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                //output
                grad_sample_values.packed_accessor32<float,2,torch::RestrictPtrTraits>()
            );

    }else if(val_dim==2){
         VolumeRenderingGPU::sum_over_each_ray_backward_gpu<2><<<blocks, BLOCK_SIZE>>>(
                nr_rays,
                ray_samples_packed.max_nr_samples, //useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
                ray_samples_packed.ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.rays_have_equal_nr_of_samples,
                ray_samples_packed.fixed_nr_of_samples_per_ray,
                grad_values_sum_per_ray.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                grad_values_sum_per_sample.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                sample_values.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                //output
                grad_sample_values.packed_accessor32<float,2,torch::RestrictPtrTraits>()
            );
    }else if(val_dim==3){
        VolumeRenderingGPU::sum_over_each_ray_backward_gpu<3><<<blocks, BLOCK_SIZE>>>(
                nr_rays,
                ray_samples_packed.max_nr_samples, //useful for checking if the ray has samples higher the the max_nr_samples in which case we don't integrate
                ray_samples_packed.ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.rays_have_equal_nr_of_samples,
                ray_samples_packed.fixed_nr_of_samples_per_ray,
                grad_values_sum_per_ray.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                grad_values_sum_per_sample.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                sample_values.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                //output
                grad_sample_values.packed_accessor32<float,2,torch::RestrictPtrTraits>()
            );
    }else{
        LOG(FATAL) <<"Val dim not impelmented yet";
    }


    return grad_sample_values;

}




