#pragma once


#include <torch/torch.h>
#include "permuto_sdf/helper_math.h"

// //Add this header after we add all cuda stuff because we need the profiler to have cudaDeviceSyncronize defined
// #define ENABLE_CUDA_PROFILING 1
// #include "Profiler.h" 


#include "permuto_sdf/pcg32.h"



#define BLOCK_SIZE 256

namespace RaySamplerGPU{

inline constexpr __device__ uint32_t MAX_STEPS() { return 2048*2; } // finest number of steps per unit length

__device__ float clamp_min(float x, float a ){
  return max(a, x);
}

__device__ float clamp_gpu(float x, float a, float b){
  return max(a, min(b, x));
}

__device__ float map_range_gpu(const float input, const float input_start,const float input_end, const float output_start, const float output_end) {
    //we clamp the input between the start and the end
    float input_clamped=clamp_gpu(input, input_start, input_end);
    return output_start + ((output_end - output_start) / (input_end - input_start)) * (input_clamped - input_start);
}


__global__ void 
compute_samples_bg_gpu(
    const int nr_rays,
    const int nr_samples_per_ray,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_origins,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_dirs,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_t_exit,
    const float sphere_radius,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> sphere_center_tensor,
    pcg32 rng,
    const bool randomize_position,
    const bool contract_3d_samples,
    //output
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> samples_3d,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> samples_4d,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> samples_dirs,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_z,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_dt,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_fixed_dt,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> ray_start_end_idx
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_rays){ //don't go out of bounds
        return;
    }

    float t_exit=ray_t_exit[idx][0];
    float3 ray_dir;
    ray_dir.x=ray_dirs[idx][0];
    ray_dir.y=ray_dirs[idx][1];
    ray_dir.z=ray_dirs[idx][2];
    float3 ray_origin;
    ray_origin.x=ray_origins[idx][0];
    ray_origin.y=ray_origins[idx][1];
    ray_origin.z=ray_origins[idx][2];
    float3 sphere_center;
    sphere_center.x=sphere_center_tensor[0];
    sphere_center.y=sphere_center_tensor[1];
    sphere_center.z=sphere_center_tensor[2];


    //get a t for each sample between 0(infinity) and 1
    float min_t=1e-3; //min_t is not exactly 0 because mapping 0 to infinty means 1/0 so we rather do 1/min_t which is pretty close to infinity
    float t_between_samples=(1.0-min_t)/(nr_samples_per_ray-1); //if we have 2 samples, we have only 1 gab in between and the distance is therefore just 1-min_t

    for (int i=0; i<nr_samples_per_ray; i++){
        //the first sample starts at t=1 (sphere_surface) and they go towards min_t(infinity)
        float t_sample=1.0-i*t_between_samples;
        //TODO randomize
        if(randomize_position){
            rng.advance(idx*nr_samples_per_ray);
            //move the samples +- t_between_samples/2
            float rand=rng.next_float();
            float rand_mov=t_between_samples*rand-t_between_samples/2.0; //moves between [-half_dist, half_dist]
            t_sample+=rand_mov; 
        }
        t_sample=clamp(t_sample, min_t, 1.0); //clamp because randomize can make it go further than the allowed range

        //map 1 towards the exit of the bounding primitive and 0 to infinity
        float z_sample=t_exit/t_sample;
        samples_z[idx][i]=z_sample;

        //store the 3d point
        float3 sample_3d = ray_origin + z_sample * ray_dir;
        if(contract_3d_samples){ //similar to mipnerf but the radius we take is not 1 but its the sphere radius
            // float t_sample_sphere_radius_0=map_range_gpu(t_sample, 1.0, 0.0, sphere_radius, 0.0); //t_sample goes from 1(exit from the sphere) towards 0. We make it go from sphere radius towards 0
            float t_sample_sphere_radius_0=t_sample*sphere_radius;  //t_sample goes from 1(exit from the sphere) towards 0. We make it go from sphere radius towards 0
            float point_dist_from_center=length(sample_3d);
            float3 point_direction_from_center=sample_3d / point_dist_from_center; 
            //when the point_dist_from_center is infinity we have 2*sphere_radius-0 so we have 2*sphere_radius*point_direction_from_center;
            //when the point_dist_from_center is sphere_radius we have 2*sphere_radius - sphere radius 
            sample_3d = (2*sphere_radius-t_sample_sphere_radius_0) * point_direction_from_center;

        }
        samples_3d[idx][i][0]=sample_3d.x;
        samples_3d[idx][i][1]=sample_3d.y;
        samples_3d[idx][i][2]=sample_3d.z;

        // make a 4d point similar to nerf++
        // the 4d representation is (dir, sphere_radius/r) #THE direction is in the frame of the sphere!
        // the r is the distance from the center of the sphere
        // the direction should be not the direction of the camera but the direction starting from the center of the sphere
        float3 points_in_sphere_frame=sample_3d-sphere_center;
        float3 direction_from_center=normalize(points_in_sphere_frame); //make it an actual direction by normalizing
        float distance_from_center=length(points_in_sphere_frame);  //in range radius to infinity
        float t_10_sphere=sphere_radius/clamp_min(distance_from_center, 1e-6); //in range [1,0]

        //store the 4d sample
        // ray_samples_bg_4d=torch.cat([direction_from_center, t_10_sphere.view(nr_rays, self.N_samples_bg, 1)], -1)
        samples_4d[idx][i][0]=direction_from_center.x;
        samples_4d[idx][i][1]=direction_from_center.y;
        samples_4d[idx][i][2]=direction_from_center.z;
        samples_4d[idx][i][3]=t_10_sphere;

        //store the dir
        samples_dirs[idx][i][0]=ray_dir.x;
        samples_dirs[idx][i][1]=ray_dir.y;
        samples_dirs[idx][i][2]=ray_dir.z;

    }

    //dt
    for (int i=0; i<nr_samples_per_ray-1; i++){
        float cur_z=samples_z[idx][i];
        float next_z=samples_z[idx][i+1];
        float dt=next_z-cur_z;
        samples_dt[idx][i]=dt;
    }
    //last dt
    samples_dt[idx][nr_samples_per_ray-1]=1e10;


    //per ray quantities
    ray_fixed_dt[idx][0]=0; //the samples have different dt for each sample so we set this to 0
    //ray_start_end_idx
    ray_start_end_idx[idx][0]=idx*nr_samples_per_ray;
    ray_start_end_idx[idx][1]=idx*nr_samples_per_ray+nr_samples_per_ray;


}



__global__ void 
compute_samples_fg_gpu(
    const int nr_rays,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_origins,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_dirs,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_t_entry,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_t_exit,
    const float sphere_radius,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> sphere_center_tensor,
    const float min_dist_between_samples,
    const int max_nr_samples_per_ray,
    const int max_nr_samples,
    pcg32 rng,
    const bool jitter_samples,
    //output
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_pos,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_dirs,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_z,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_dt,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_fixed_dt,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> ray_start_end_idx,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> cur_nr_samples
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_rays){ //don't go out of bounds
        return;
    }

    // float t_exit=ray_t_exit[idx][0];
    float t_start=ray_t_entry[idx][0];
    float t_exit=ray_t_exit[idx][0];
    float3 ray_dir;
    ray_dir.x=ray_dirs[idx][0];
    ray_dir.y=ray_dirs[idx][1];
    ray_dir.z=ray_dirs[idx][2];
    float3 ray_origin;
    ray_origin.x=ray_origins[idx][0];
    ray_origin.y=ray_origins[idx][1];
    ray_origin.z=ray_origins[idx][2];
    float3 sphere_center;
    sphere_center.x=sphere_center_tensor[0];
    sphere_center.y=sphere_center_tensor[1];
    sphere_center.z=sphere_center_tensor[2];

    float eps=1e-6; // float32 only has 7 decimal digits precision

    float distance_through_occupied_space=t_exit-t_start; //this can be 0 in the case tha the ray doesn't intersect with the bounding primitive

    //now we have a maximum distance that we traverse through occupied space. 
    //we also have a min distance between samples, so we can calculate how many samples we need
    int nr_samples_to_create=distance_through_occupied_space/min_dist_between_samples;

    //this nr of samples can be quite large if the occupied distance is big, so we clamp the nr of samples
    nr_samples_to_create=clamp(nr_samples_to_create, 0, max_nr_samples_per_ray);

    //recalculate the dist between samples given the new nr of samples
    float dist_between_samples=distance_through_occupied_space/nr_samples_to_create;


    //if we have samples to create we create them
    //we also don't create anything if we have only 1 sample. Only 1 sample creates lots of probles with the cdf which has only one value of 0 and therefore binary search runs forever
    if (nr_samples_to_create>1 && distance_through_occupied_space>eps){
        //go again through the occupancy grid and create the samples, we assume we have enough space for them
        //first do some bookeeping for this ray, like the nr of samples, so we know where to write in the packed tensor
        int indx_start_sample=atomicAdd(&cur_nr_samples[0],nr_samples_to_create);
        ray_start_end_idx[idx][0]=indx_start_sample;
        // ray_start_end_idx[idx][1]=indx_start_sample+nr_samples_to_create-1; //we substract 1 because if we create for example 2 samples we want the indices to be the start idx to be 0 and the end idx to be 1. So the start and end idx point directly at the first and last sampler
        ray_start_end_idx[idx][1]=indx_start_sample+nr_samples_to_create; 
        ray_fixed_dt[idx][0]=dist_between_samples;
        //if we are about to create more samples that we can actually store, just exit this thread
        //however we still keep writing into ray_start_end_idx because we want to be able later to filter those rays out that actually have no samples and not process them or not volumetrically integrate them
        if( (indx_start_sample+nr_samples_to_create)>max_nr_samples){
            return;
        }


        float t=t_start; //cur_t
        int nr_steps=0;

        //jutter just the begginign so the samples are all at the same distance from each other and therefore the dt is all the same
        float rand_mov=0;
        if(jitter_samples){
            rng.advance(idx); //since all the threads start with the same seed, we need to advance this thread so it gets other numbers different than the otehrs
            t=t+dist_between_samples*rng.next_float();;
        }


        int nr_samples_created=0;
        while (t<t_exit &&nr_steps<MAX_STEPS()) {
            t=clamp(t, t_start, t_exit);
            float3 pos = ray_origin+t*ray_dir;
            
            //we also check that we don't create more samples because sometimes it can happen to create one or more less depending on floating point errors
            if (nr_samples_created<nr_samples_to_create){ //we advance the t just a bit and create samples
                //store positions
                samples_pos[indx_start_sample+nr_samples_created][0]=pos.x;
                samples_pos[indx_start_sample+nr_samples_created][1]=pos.y;
                samples_pos[indx_start_sample+nr_samples_created][2]=pos.z;
                //store dirs
                samples_dirs[indx_start_sample+nr_samples_created][0]=ray_dir.x;
                samples_dirs[indx_start_sample+nr_samples_created][1]=ray_dir.y;
                samples_dirs[indx_start_sample+nr_samples_created][2]=ray_dir.z;
                //store z
                samples_z[indx_start_sample+nr_samples_created][0]=t;
                samples_dt[indx_start_sample+nr_samples_created][0]=dist_between_samples;
                //go to next sample
                // float rand_mov=0;
                // if(jitter_samples){
                //     rng.advance(idx); //since all the threads start with the same seed, we need to advance this thread so it gets other numbers different than the otehrs
                //     float rand=rng.next_float();
                //     rand_mov=dist_between_samples*rand-dist_between_samples/2.0; //moves between [-half_dist, half_dist]
                // }
                // t+=dist_between_samples+rand_mov;
                t+=dist_between_samples;
                // t=clamp(t, t_start, t_exit);
                nr_samples_created+=1;
            }
            nr_steps+=1;
        }
        //better to store exactly the nr of samples we created
        // ray_start_end_idx[idx][0]=indx_start_sample;
        // ray_start_end_idx[idx][1]=indx_start_sample+nr_samples_created;

        //the last sample on the ray doesn't necesserally need to have a dt= dist_between_samples. It can have a lower dt if it's very close to the border
        float remaining_dist_until_border=t_exit-samples_z[indx_start_sample+nr_samples_created-1][0];
        samples_dt[indx_start_sample+nr_samples_created-1][0]=clamp(remaining_dist_until_border, 0.0, dist_between_samples);
        // samples_dt[indx_start_sample+nr_samples_created-1][0]=remaining_dist_until_border;


        //if we still have some more samples to create, we just set them to zero
        for (int i=nr_samples_created; i<nr_samples_to_create; i++){
            //store positions
            samples_pos[indx_start_sample+i][0]=0;
            samples_pos[indx_start_sample+i][1]=0;
            samples_pos[indx_start_sample+i][2]=0;
            //store dirs
            samples_dirs[indx_start_sample+i][0]=0;
            samples_dirs[indx_start_sample+i][1]=0;
            samples_dirs[indx_start_sample+i][2]=0;
            //store z
            samples_z[indx_start_sample+i][0]=-1; //just a sentinel value that we can easily detect in the volumetric rendering and discard these samples
            samples_dt[indx_start_sample+i][0]=0;
        }

        //if we create less samples than what we commited to create we just update to the new quantity so we know that end_idx points at the last sample of the ray
        // ray_start_end_idx[idx][1]=indx_start_sample+nr_samples_created-1;
        ray_start_end_idx[idx][1]=indx_start_sample+nr_samples_created;

        //if we create only 1 sample, then we discard this ray since the cdf will just have a value of 0 and therfore binary search will run forever
        //we also discard rays with 2 or less because it really doesnt make sense to integrate just 2 samples
        if(nr_samples_created<=2){
            ray_fixed_dt[idx][0]=0;
            ray_start_end_idx[idx][0]=0;
            ray_start_end_idx[idx][1]=0;
        }


    }else{
        //this ray passes onyl though unocupied space
        //we set the ray quantities to 0 and there is nothing to set in the per_sample quantities because we have no samples
        ray_fixed_dt[idx][0]=0;
        ray_start_end_idx[idx][0]=0;
        ray_start_end_idx[idx][1]=0;

        
    }





}



}//namespace occupancy grid








