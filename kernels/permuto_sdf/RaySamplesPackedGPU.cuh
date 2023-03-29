#pragma once


#include <torch/torch.h>
#include "permuto_sdf/helper_math.h"



#define BLOCK_SIZE 256

namespace RaySamplesPackedGPU{



__global__ void 
compact_to_valid_samples_gpu(
    const int nr_rays,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_pos,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_pos_4d,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_dirs,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_z,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_dt,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_sdf,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_fixed_dt,
    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> ray_start_end_idx,
    //output
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> out_samples_pos,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> out_samples_pos_4d,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> out_samples_dirs,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> out_samples_z,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> out_samples_dt,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> out_samples_sdf,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> out_ray_fixed_dt,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> out_ray_start_end_idx,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> out_cur_nr_samples
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_rays){ //don't go out of bounds
        return;
    }


    //get the indexes where this ray starts and end 
    int in_idx_start=ray_start_end_idx[idx][0];
    int in_idx_end=ray_start_end_idx[idx][1];
    int nr_samples_per_ray=in_idx_end-in_idx_start; 

    //allocate samples for out ray 
    int out_idx_start=atomicAdd(&out_cur_nr_samples[0], nr_samples_per_ray);
    //copy all the samples
    for (int i=0; i<nr_samples_per_ray; i++){
        //samples_pos
        out_samples_pos[out_idx_start+i][0]=samples_pos[in_idx_start+i][0];
        out_samples_pos[out_idx_start+i][1]=samples_pos[in_idx_start+i][1];
        out_samples_pos[out_idx_start+i][2]=samples_pos[in_idx_start+i][2];
        //samples_pos_4d
        out_samples_pos_4d[out_idx_start+i][0]=samples_pos_4d[in_idx_start+i][0];
        out_samples_pos_4d[out_idx_start+i][1]=samples_pos_4d[in_idx_start+i][1];
        out_samples_pos_4d[out_idx_start+i][2]=samples_pos_4d[in_idx_start+i][2];
        out_samples_pos_4d[out_idx_start+i][3]=samples_pos_4d[in_idx_start+i][3];
        //samples_dirs
        out_samples_dirs[out_idx_start+i][0]=samples_dirs[in_idx_start+i][0];
        out_samples_dirs[out_idx_start+i][1]=samples_dirs[in_idx_start+i][1];
        out_samples_dirs[out_idx_start+i][2]=samples_dirs[in_idx_start+i][2];
        //samples_z
        out_samples_z[out_idx_start+i][0]=samples_z[in_idx_start+i][0];
        //samples_dt
        out_samples_dt[out_idx_start+i][0]=samples_dt[in_idx_start+i][0];
        //samples_sdf
        out_samples_sdf[out_idx_start+i][0]=samples_sdf[in_idx_start+i][0];
    }
    //per ray values
    out_ray_fixed_dt[idx][0]=ray_fixed_dt[idx][0];
    out_ray_start_end_idx[idx][0]=out_idx_start;
    out_ray_start_end_idx[idx][1]=out_idx_start+nr_samples_per_ray;



}





}//namespace 








