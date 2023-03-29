#include "permuto_sdf/RaySampler.cuh"

//c++
// #include <string>

#include "UtilsPytorch.h" //contains torch so it has to be added BEFORE any other include because the other ones might include loguru which gets screwed up if torch was included before it

//my stuff
#include "permuto_sdf/RaySamplerGPU.cuh"



using torch::Tensor;
using namespace radu::utils;



template <typename T>
T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

pcg32 RaySampler::m_rng;


//CPU code that calls the kernels
RaySampler::RaySampler()
    {


}

RaySampler::~RaySampler(){
}


RaySamplesPacked RaySampler::compute_samples_bg(const torch::Tensor& ray_origins, const torch::Tensor& ray_dirs, const torch::Tensor& ray_t_exit, const int nr_samples_per_ray, const float sphere_radius, const torch::Tensor& sphere_center, const bool randomize_position, const bool contract_3d_samples){

    CHECK(ray_origins.dim()==2) << "ray_origins should have dim 2 correspondin to nr_raysx3. However it has sizes" << ray_origins.sizes();
    CHECK(ray_dirs.dim()==2) << "ray_dirs should have dim 2 correspondin to nr_raysx3. However it has sizes" << ray_dirs.sizes();
    CHECK(ray_t_exit.dim()==2) << "ray_t_exit should have dim 2 correspondin to nr_raysx1. However it has sizes" << ray_t_exit.sizes();

    int nr_rays=ray_origins.size(0);

    int nr_samples_maximum=nr_rays*nr_samples_per_ray;
    RaySamplesPacked ray_samples_packed(nr_rays, nr_samples_maximum);
    ray_samples_packed.rays_have_equal_nr_of_samples=true;
    ray_samples_packed.fixed_nr_of_samples_per_ray=nr_samples_per_ray;

    //view them a bit different because it's easier to fill them
    ray_samples_packed.samples_z = ray_samples_packed.samples_z.view({ nr_rays, nr_samples_per_ray });
    ray_samples_packed.samples_dt = ray_samples_packed.samples_dt.view({ nr_rays, nr_samples_per_ray });
    ray_samples_packed.samples_pos = ray_samples_packed.samples_pos.view({ nr_rays, nr_samples_per_ray,3 });
    ray_samples_packed.samples_pos_4d = ray_samples_packed.samples_pos_4d.view({ nr_rays, nr_samples_per_ray,4 });
    ray_samples_packed.samples_dirs = ray_samples_packed.samples_dirs.view({ nr_rays, nr_samples_per_ray,3 });


    const dim3 blocks = { (unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1 }; 


    RaySamplerGPU::compute_samples_bg_gpu<<<blocks, BLOCK_SIZE>>>(
                nr_rays,
                nr_samples_per_ray,
                ray_origins.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_dirs.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_t_exit.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                sphere_radius,
                sphere_center.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                m_rng,
                randomize_position,
                contract_3d_samples,
                //output
                ray_samples_packed.samples_pos.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                ray_samples_packed.samples_pos_4d.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                ray_samples_packed.samples_dirs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                ray_samples_packed.samples_z.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.samples_dt.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.ray_fixed_dt.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>()
            );

    if(randomize_position){
        m_rng.advance();
    }


    ray_samples_packed.cur_nr_samples.fill_(nr_rays*nr_samples_per_ray);


    ray_samples_packed.samples_z = ray_samples_packed.samples_z.view({ nr_samples_maximum,1 });
    ray_samples_packed.samples_dt = ray_samples_packed.samples_dt.view({ nr_samples_maximum,1 });
    ray_samples_packed.samples_pos = ray_samples_packed.samples_pos.view({ nr_samples_maximum,3 });
    ray_samples_packed.samples_pos_4d = ray_samples_packed.samples_pos_4d.view({ nr_samples_maximum,4 });
    ray_samples_packed.samples_dirs = ray_samples_packed.samples_dirs.view({ nr_samples_maximum,3 });

    return ray_samples_packed;
    // return std::make_tuple(z_vals, samples_3d, samples_4d);

    // return grid_points;

}


RaySamplesPacked RaySampler::compute_samples_fg(const torch::Tensor& ray_origins, const torch::Tensor& ray_dirs, const torch::Tensor& ray_t_entry, const torch::Tensor& ray_t_exit, const float min_dist_between_samples, const int max_nr_samples_per_ray, const float sphere_radius, const torch::Tensor& sphere_center, const bool randomize_position){

    CHECK(ray_origins.dim()==2) << "ray_origins should have dim 2 correspondin to nr_raysx3. However it has sizes" << ray_origins.sizes();
    CHECK(ray_dirs.dim()==2) << "ray_dirs should have dim 2 correspondin to nr_raysx3. However it has sizes" << ray_dirs.sizes();
    CHECK(ray_t_entry.dim()==2) << "ray_t_entry should have dim 2 correspondin to nr_raysx1. However it has sizes" << ray_t_entry.sizes();
    CHECK(ray_t_exit.dim()==2) << "ray_t_exit should have dim 2 correspondin to nr_raysx1. However it has sizes" << ray_t_exit.sizes();

    int nr_rays=ray_origins.size(0);

    int nr_samples_maximum=nr_rays*max_nr_samples_per_ray;
    RaySamplesPacked ray_samples_packed(nr_rays, nr_samples_maximum);


    const dim3 blocks = { (unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1 }; 

    RaySamplerGPU::compute_samples_fg_gpu<<<blocks, BLOCK_SIZE>>>(
                nr_rays,
                ray_origins.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_dirs.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_t_entry.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_t_exit.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                sphere_radius,
                sphere_center.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                min_dist_between_samples,
                max_nr_samples_per_ray,
                ray_samples_packed.max_nr_samples,
                m_rng,
                randomize_position,
                //output
                ray_samples_packed.samples_pos.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.samples_dirs.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.samples_z.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.samples_dt.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.ray_fixed_dt.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.cur_nr_samples.packed_accessor32<int,1,torch::RestrictPtrTraits>()
            );

    if(randomize_position){
        m_rng.advance();
    }


    

    return ray_samples_packed;


}


