#pragma once


// #include "hash_sdf/jitify_helper/jitify_helper.cuh"
#include <torch/torch.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cuda_runtime_api.h>
// #include "device_launch_parameters.h" //needed for threadIdx and blockDim 
// #include <torch/torch.h>
#include "hash_sdf/helper_math.h"

// //Add this header after we add all cuda stuff because we need the profiler to have cudaDeviceSyncronize defined
// #define ENABLE_CUDA_PROFILING 1
// #include "Profiler.h" 

//matrices
#include "hash_sdf/mat3.h"
#include "hash_sdf/mat4.h"

#include "hash_sdf/pcg32.h"



#define BLOCK_SIZE 256

namespace OccupancyGridGPU{

//from https://github.com/NVlabs/instant-ngp/blob/e1d33a42a4de0b24237685f2ebdc07bcef1ecae9/src/testbed_nerf.cu
inline constexpr __device__ uint32_t MAX_STEPS() { return 2048*2; } // finest number of steps per unit length
// inline constexpr __device__ float SQRT3() { return 1.73205080757f; }
// inline constexpr __device__ float STEPSIZE() { return (SQRT3() / MAX_STEPS()); } // for nerf raymarch
// inline constexpr __device__ float MIN_STEPSIZE() { return STEPSIZE(); }
// Maximum step size is the width of the coarsest gridsize cell.
// inline constexpr __device__ float MAX_STEPSIZE() { return STEPSIZE() * (1<<(NERF_CASCADES()-1)) * NERF_STEPS() / NERF_GRIDSIZE(); }

//page 86/146 from https://core.ac.uk/download/pdf/85209106.pdf
// __device__ float barycentric_to_c2_continous(const float x) {

//     float y= 6*powf(x,5) - 15*powf(x,4) +10*powf(x,3);

//     return y;
// }
// __device__ float barycentric_to_c1_continous(const float x) {

//     float y= 3*powf(x,2) - 2*powf(x,3);

//     return y;
// }
// __device__ float barycentric_to_c1_continous_back(const float x) {

//     float y= 6*x - 6*powf(x,2);

//     return y;
// }


template <typename T> __device__ void inline swap(T a, T b){
    T c(a); a=b; b=c;
}




//from https://github.com/NVlabs/tiny-cuda-nn/blob/a56341f3dd02c709d36aaa6406d94b5a80cf2d94/include/tiny-cuda-nn/common_device.h
// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__host__ __device__ inline uint32_t expand_bits(uint32_t v) {
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}




// Calculates a 30-bit Morton code for the
// given 3D idx located within the unit cube [0,nr_voxels_per_dim-1].
__host__ __device__ inline uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z) {
	uint32_t xx = expand_bits(x);
	uint32_t yy = expand_bits(y);
	uint32_t zz = expand_bits(z);
	return xx | (yy << 1) | (zz << 2);
}
__host__ __device__ inline uint32_t morton3D_clamped(int x, int y,  int z, const int nr_voxels_per_dim) {
    x=clamp(x,0,nr_voxels_per_dim-1);
    y=clamp(y,0,nr_voxels_per_dim-1);
    z=clamp(z,0,nr_voxels_per_dim-1);
    return morton3D(x,y,z);
}

 
__host__ __device__ inline uint32_t morton3D_invert(uint32_t x) {
	x = x               & 0x49249249;
	x = (x | (x >> 2))  & 0xc30c30c3;
	x = (x | (x >> 4))  & 0x0f00f00f;
	x = (x | (x >> 8))  & 0xff0000ff;
	x = (x | (x >> 16)) & 0x0000ffff;
	return x;
}

__host__ __device__ inline uint64_t expand_bits(uint64_t w)  {
	w &=                0x00000000001fffff;
	w = (w | w << 32) & 0x001f00000000ffff;
	w = (w | w << 16) & 0x001f0000ff0000ff;
	w = (w | w <<  8) & 0x010f00f00f00f00f;
	w = (w | w <<  4) & 0x10c30c30c30c30c3;
	w = (w | w <<  2) & 0x1249249249249249;
	return w;
}

//https://forums.developer.nvidia.com/t/sign-function/18375/4
__device__ int sign(float x){ 
	int t = x<0 ? -1 : 0;
	return x > 0 ? 1 : t;
}


// inline __host__ __device__ float calc_dt(float t) {
// 	return clamp(t, MIN_STEPSIZE(), 1e32);
// }

//https://github.com/NVlabs/instant-ngp/blob/master/src/testbed_nerf.cu
inline __device__ float distance_to_next_voxel(float3 pos, const float3& dir, const float3& idir, const int nr_voxels_per_dim) { // dda like step
    // pos+=0.5;
    pos=nr_voxels_per_dim*pos;
    // printf("inDirsto_nexTvoxel pos is %f,%f,%f \n", pos.x, pos.y, pos.z);
	float tx = (floorf(pos.x + 0.5f + 0.5f * sign(dir.x)) - pos.x) * idir.x;
	float ty = (floorf(pos.y + 0.5f + 0.5f * sign(dir.y)) - pos.y) * idir.y;
	float tz = (floorf(pos.z + 0.5f + 0.5f * sign(dir.z)) - pos.z) * idir.z;
	// float t = min(min(tx, ty), tz);
	float t = min(min(abs(tx), abs(ty)), abs(tz));
    // printf("delta t is %f, tx,ty,tz are %f,%f,%f \n ",t, tx,ty,tz);

    t=fmaxf(t/nr_voxels_per_dim, 0.0f);

	return t;
}

// inline __device__ float advance_to_next_voxel(float t, const float3& pos, const float3& dir, const float3& idir, const int nr_voxels_per_dim) {
// 	// Analytic stepping by a multiple of dt. Make empty space unequal to non-empty space
// 	// due to the different stepping.
// 	// float dt = calc_dt(t, cone_angle);
// 	// return t + ceilf(fmaxf(distance_to_next_voxel(pos, dir, idir, res) / dt, 0.5f)) * dt;

// 	// Regular stepping (may be slower but matches non-empty space)
// 	float t_target = t + distance_to_next_voxel(pos, dir, idir, nr_voxels_per_dim);
// 	do {
// 		t += calc_dt(t);
// 	} while (t < t_target);
// 	return t;
// }




__host__ __device__ inline float3 lin_idx_to_3D(const uint32_t idx, const int nr_voxels_per_dim, const float grid_extent, const float3 grid_translation, const bool get_center_of_voxel) {
    float x = morton3D_invert(idx>>0); 
	float y = morton3D_invert(idx>>1); //right bit shift divide by 2 and floors
	float z = morton3D_invert(idx>>2);
    //now the xyz are in range [0,nr_voxels_per_dim-1] so it just does a translation from 1d idx to 3D idx

    // if(idx<=10){
    //     printf("In idx_to_3d idx is %d,  x,y,z is %f, %f, %f\n", idx, x, y, z);
    // }

    x=x/nr_voxels_per_dim;
    y=y/nr_voxels_per_dim;
    z=z/nr_voxels_per_dim;
    //now xyz is in range [0,1-voxel_size]

    //shift so that it doesnt start at 0,0,0 but rather the center of the grid is at 0.0.0
    x=x-0.5;
    y=y-0.5;
    z=z-0.5;


    if(get_center_of_voxel){
        //we want to get the center of the voxel so we shift by half of the voxel_size
        float voxel_size=1.0/nr_voxels_per_dim; //we divicde 1 by the voxel per dim because now we have xyz in normalized coordiantes so between 0 and 1-voxelssize
        float half_voxel_size=voxel_size/2;
        x+=half_voxel_size;
        y+=half_voxel_size;
        z+=half_voxel_size;
    }
    //now we have either the center points of the voxels or the lower left corner and it has extent of 1

    //we apply the grid extent because now we have the grid centered aroudn the origin
    x=x*grid_extent;
    y=y*grid_extent;
    z=z*grid_extent;

    //now we apply the translation
    x=x+grid_translation.x;
    y=y+grid_translation.y;
    z=z+grid_translation.z;



    float3 pos=make_float3(x,y,z);

    return pos;
}

//position in some world coordinates
__host__ __device__ inline int pos_to_lin_idx(float3 pos, const int nr_voxels_per_dim, const float grid_extent, const float3 grid_translation, const bool get_center_of_voxel) {
    //we go in reverse order of the lin_idx_to_3D

    //remove translation
    pos.x=pos.x-grid_translation.x;
    pos.y=pos.y-grid_translation.y;
    pos.z=pos.z-grid_translation.z;

    //we apply the grid extent because now we have the grid centered aroudn the origin
    pos.x=pos.x/grid_extent;
    pos.y=pos.y/grid_extent;
    pos.z=pos.z/grid_extent;


    if(get_center_of_voxel){
        //we want to get the center of the voxel so we shift by half of the voxel_size
        float voxel_size=1.0/nr_voxels_per_dim; //we divicde 1 by the voxel per dim because now we have xyz in normalized coordiantes so between 0 and 1-voxelssize
        float half_voxel_size=voxel_size/2;
        pos.x-=half_voxel_size;
        pos.y-=half_voxel_size;
        pos.z-=half_voxel_size;
    }

    //shift so that it doesnt start at 0,0,0 but rather the center of the grid is at 0.0.0
    pos.x=pos.x+0.5;
    pos.y=pos.y+0.5;
    pos.z=pos.z+0.5;

    pos.x=pos.x*nr_voxels_per_dim;
    pos.y=pos.y*nr_voxels_per_dim;
    pos.z=pos.z*nr_voxels_per_dim;

    // printf("pos just before morton is %f,%f,%f \n", pos.x, pos.y, pos.z);

    int idx= morton3D(pos.x, pos.y, pos.z);

    return idx;
}


__global__ void 
compute_grid_points_gpu(
    const int nr_voxels,
    const int nr_voxels_per_dim,
    const float grid_extent,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> grid_translation_tensor,
    pcg32 rng,
    const bool randomize_position,
    //output
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> center_points
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_voxels){ //don't go out of bounds
        return;
    }

    float3 grid_translation=make_float3(grid_translation_tensor[0], grid_translation_tensor[1], grid_translation_tensor[2]);


    // uint32_t x = morton3D_invert(idx>>0); 
	// uint32_t y = morton3D_invert(idx>>1); //right bit shift divide by 2 and floors
	// uint32_t z = morton3D_invert(idx>>2);

    float3 pos3d=lin_idx_to_3D(idx, nr_voxels_per_dim, grid_extent, grid_translation, true);

    if(randomize_position){
        float voxel_size=grid_extent/nr_voxels_per_dim;
        float half_voxel_size=voxel_size/2.0;

        float rand;
        float mov;

        rng.advance(idx*3); //since all the threads start with the same seed, we need to advance this thread so it gets other numbers different than the otehrs
        //x
        rand=rng.next_float();//random in range [0,1]
        mov=voxel_size*rand-half_voxel_size;
        pos3d.x+=mov;
        //y
        rand=rng.next_float();//random in range [0,1]
        mov=voxel_size*rand-half_voxel_size;
        pos3d.y+=mov;
        //z
        rand=rng.next_float();//random in range [0,1]
        mov=voxel_size*rand-half_voxel_size;
        pos3d.z+=mov;

    }

    center_points[idx][0]=pos3d.x; 
    center_points[idx][1]=pos3d.y; 
    center_points[idx][2]=pos3d.z; 

}

__global__ void 
compute_random_sample_of_grid_points_gpu(
    const int nr_voxels_to_select,
    const int nr_voxels_per_dim,
    const float grid_extent,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> grid_translation_tensor,
    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> point_indices,
    pcg32 rng,
    const bool randomize_position,
    //output
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> center_points
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_voxels_to_select){ //don't go out of bounds
        return;
    }

    float3 grid_translation=make_float3(grid_translation_tensor[0], grid_translation_tensor[1], grid_translation_tensor[2]);

    int idx_voxel=point_indices[idx];


    float3 pos3d=lin_idx_to_3D(idx_voxel, nr_voxels_per_dim, grid_extent, grid_translation, true);

    if(randomize_position){
        float voxel_size=grid_extent/nr_voxels_per_dim;
        float half_voxel_size=voxel_size/2.0;

        float rand;
        float mov;

        rng.advance(idx*3); //since all the threads start with the same seed, we need to advance this thread so it gets other numbers different than the otehrs
        //x
        rand=rng.next_float();//random in range [0,1]
        mov=voxel_size*rand-half_voxel_size;
        pos3d.x+=mov;
        //y
        rand=rng.next_float();//random in range [0,1]
        mov=voxel_size*rand-half_voxel_size;
        pos3d.y+=mov;
        //z
        rand=rng.next_float();//random in range [0,1]
        mov=voxel_size*rand-half_voxel_size;
        pos3d.z+=mov;

    }

    center_points[idx][0]=pos3d.x; 
    center_points[idx][1]=pos3d.y; 
    center_points[idx][2]=pos3d.z; 

}

__global__ void 
update_with_density_gpu(
    const int nr_voxels,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> density_tensor,
    const int nr_voxels_per_dim,
    const float decay,
    const float occupancy_tresh,
    // const bool check_neighbours_density,
    //output
    // torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grid_tensor
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> grid_values_tensor,
    torch::PackedTensorAccessor32<bool,1,torch::RestrictPtrTraits> grid_occupancy_tensor
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_voxels){ //don't go out of bounds
        return;
    }

    //the index here is already in morton order since the points were generated in morton order 



    // int x_idx = morton3D_invert(idx>>0); 
	// int y_idx = morton3D_invert(idx>>1); //right bit shift divide by 2 and floors
	// int z_idx = morton3D_invert(idx>>2);

    // if(idx<=10){
    //     printf("idx is %d,  x,y,z is %d, %d, %d\n", idx,x_idx, y_idx, z_idx);
    // }
    
    // float old_density=grid_tensor[x_idx][y_idx][z_idx];
    float old_density=grid_values_tensor[idx];


    old_density=old_density*decay; //decay
    float new_density=density_tensor[idx][0];



    // if (check_neighbours_density){
    //     //get the maximum density in neighbourhood
    //     //http://www.alecjacobson.com/weblog/?p=1425
    //     //assumes the order is zyx where x is the fastest changing dimension and z the slowest
    //     // int x = idx % nr_voxels_per_dim;
    //     // int y = (idx - x)/nr_voxels_per_dim % nr_voxels_per_dim;
    //     // int z = ((idx - x)/nr_voxels_per_dim-y)/ nr_voxels_per_dim;
    //     int x = morton3D_invert(idx>>0); 
    //     int y = morton3D_invert(idx>>1); //right bit shift divide by 2 and floors
    //     int z = morton3D_invert(idx>>2);
    //     //switch from zyx to xyz
    //     // swap(z,x);
    //     //now the xyz are in range [0,nr_voxels_per_dim-1] so it just does a translation from 1d idx to 3D idx
    //     //calculate the indices of the 8 neighbours
    //     //back side
    //     int idx_0=morton3D_clamped(x-1, y-1, z-1,nr_voxels_per_dim);
    //     int idx_1=morton3D_clamped(x, y-1, z-1,nr_voxels_per_dim);
    //     int idx_2=morton3D_clamped(x+1, y-1, z-1,nr_voxels_per_dim);
    //     int idx_3=morton3D_clamped(x-1, y, z-1,nr_voxels_per_dim);
    //     int idx_4=morton3D_clamped(x, y, z-1,nr_voxels_per_dim);
    //     int idx_5=morton3D_clamped(x+1, y, z-1,nr_voxels_per_dim);
    //     int idx_6=morton3D_clamped(x-1, y+1, z-1,nr_voxels_per_dim);
    //     int idx_7=morton3D_clamped(x, y+1, z-1,nr_voxels_per_dim);
    //     int idx_8=morton3D_clamped(x+1, y+1, z-1,nr_voxels_per_dim);
    //     //middle slice
    //     int idx_9=morton3D_clamped(x-1, y-1, z,nr_voxels_per_dim);
    //     int idx_10=morton3D_clamped(x, y-1, z,nr_voxels_per_dim);
    //     int idx_11=morton3D_clamped(x+1, y-1, z,nr_voxels_per_dim);
    //     int idx_12=morton3D_clamped(x-1, y, z,nr_voxels_per_dim);
    //     int idx_13=morton3D_clamped(x, y, z,nr_voxels_per_dim);
    //     int idx_14=morton3D_clamped(x+1, y, z,nr_voxels_per_dim);
    //     int idx_15=morton3D_clamped(x-1, y+1, z,nr_voxels_per_dim);
    //     int idx_16=morton3D_clamped(x, y+1, z,nr_voxels_per_dim);
    //     int idx_17=morton3D_clamped(x+1, y+1, z,nr_voxels_per_dim);
    //     //front slice
    //     int idx_18=morton3D_clamped(x-1, y-1, z+1,nr_voxels_per_dim);
    //     int idx_19=morton3D_clamped(x, y-1, z+1,nr_voxels_per_dim);
    //     int idx_20=morton3D_clamped(x+1, y-1, z+1,nr_voxels_per_dim);
    //     int idx_21=morton3D_clamped(x-1, y, z+1,nr_voxels_per_dim);
    //     int idx_22=morton3D_clamped(x, y, z+1,nr_voxels_per_dim);
    //     int idx_23=morton3D_clamped(x+1, y, z+1,nr_voxels_per_dim);
    //     int idx_24=morton3D_clamped(x-1, y+1, z+1,nr_voxels_per_dim);
    //     int idx_25=morton3D_clamped(x, y+1, z+1,nr_voxels_per_dim);
    //     int idx_26=morton3D_clamped(x+1, y+1, z+1,nr_voxels_per_dim);
    //     // int idx_middle=morton3D(x, y, z);
    //     //the idx middle should be same as idx
    //     // if(idx_middle!=idx){
    //         // printf("They are not the same, idx_middle is %d and idx is %d \n", idx_middle, idx);
    //     // }

    //     //get the maximum value in neighbourhood
    //     //back side
    //     if(density_tensor[idx_0][0]>new_density) new_density=density_tensor[idx_0][0];
    //     if(density_tensor[idx_1][0]>new_density) new_density=density_tensor[idx_1][0];
    //     if(density_tensor[idx_2][0]>new_density) new_density=density_tensor[idx_2][0];
    //     if(density_tensor[idx_3][0]>new_density) new_density=density_tensor[idx_3][0];
    //     if(density_tensor[idx_4][0]>new_density) new_density=density_tensor[idx_4][0];
    //     if(density_tensor[idx_5][0]>new_density) new_density=density_tensor[idx_5][0];
    //     if(density_tensor[idx_6][0]>new_density) new_density=density_tensor[idx_6][0];
    //     if(density_tensor[idx_7][0]>new_density) new_density=density_tensor[idx_7][0];
    //     if(density_tensor[idx_8][0]>new_density) new_density=density_tensor[idx_8][0];
    //     //middle slice
    //     if(density_tensor[idx_9][0]>new_density) new_density=density_tensor[idx_9][0];
    //     if(density_tensor[idx_10][0]>new_density) new_density=density_tensor[idx_10][0];
    //     if(density_tensor[idx_11][0]>new_density) new_density=density_tensor[idx_11][0];
    //     if(density_tensor[idx_12][0]>new_density) new_density=density_tensor[idx_12][0];
    //     if(density_tensor[idx_13][0]>new_density) new_density=density_tensor[idx_13][0];
    //     if(density_tensor[idx_14][0]>new_density) new_density=density_tensor[idx_14][0];
    //     if(density_tensor[idx_15][0]>new_density) new_density=density_tensor[idx_15][0];
    //     if(density_tensor[idx_16][0]>new_density) new_density=density_tensor[idx_16][0];
    //     if(density_tensor[idx_17][0]>new_density) new_density=density_tensor[idx_17][0];
    //     //front slice
    //     if(density_tensor[idx_18][0]>new_density) new_density=density_tensor[idx_18][0];
    //     if(density_tensor[idx_19][0]>new_density) new_density=density_tensor[idx_19][0];
    //     if(density_tensor[idx_20][0]>new_density) new_density=density_tensor[idx_20][0];
    //     if(density_tensor[idx_21][0]>new_density) new_density=density_tensor[idx_21][0];
    //     if(density_tensor[idx_22][0]>new_density) new_density=density_tensor[idx_22][0];
    //     if(density_tensor[idx_23][0]>new_density) new_density=density_tensor[idx_23][0];
    //     if(density_tensor[idx_24][0]>new_density) new_density=density_tensor[idx_24][0];
    //     if(density_tensor[idx_25][0]>new_density) new_density=density_tensor[idx_25][0];
    //     if(density_tensor[idx_26][0]>new_density) new_density=density_tensor[idx_26][0];
    // }

    //update
    float updated_density=fmax(new_density, old_density);
    // grid_tensor[x_idx][y_idx][z_idx]=updated_density;


    grid_values_tensor[idx]=updated_density;
    grid_occupancy_tensor[idx]=updated_density>occupancy_tresh;

}

__global__ void 
update_with_density_random_sample_gpu(
    const int nr_points,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> density_tensor,
    const int nr_voxels_per_dim,
    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> point_indices,  
    const float decay,
    const float occupancy_tresh,
    // const bool check_neighbours_density,
    //output
    // torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grid_tensor
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> grid_values_tensor,
    torch::PackedTensorAccessor32<bool,1,torch::RestrictPtrTraits> grid_occupancy_tensor
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_points){ //don't go out of bounds
        return;
    }

    //get the idx of the voxel corresponding to this point
    int idx_voxel=point_indices[idx];
    //the index here is already in morton order since the points were generated in morton order 



    // int x_idx = morton3D_invert(idx>>0); 
	// int y_idx = morton3D_invert(idx>>1); //right bit shift divide by 2 and floors
	// int z_idx = morton3D_invert(idx>>2);

    // if(idx<=10){
    //     printf("idx is %d,  x,y,z is %d, %d, %d\n", idx,x_idx, y_idx, z_idx);
    // }
    
    // float old_density=grid_tensor[x_idx][y_idx][z_idx];
    float old_density=grid_values_tensor[idx_voxel];


    old_density=old_density*decay; //decay
    float new_density=density_tensor[idx][0];


    //update
    float updated_density=fmax(new_density, old_density);
    // grid_tensor[x_idx][y_idx][z_idx]=updated_density;


    grid_values_tensor[idx_voxel]=updated_density;
    grid_occupancy_tensor[idx_voxel]=updated_density>occupancy_tresh;

}

//https://arxiv.org/pdf/2106.10689.pdf
__device__ float logistic_density_distribution(float x, float s){ 
	float res= s*exp(-s*x)/ (  powf(   (1+exp(-s*x))  ,2)  );
    return res;
}


__global__ void 
update_with_sdf_gpu(
    const int nr_voxels,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> sdf_tensor,
    const float grid_extent,
    const int nr_voxels_per_dim,
    const float inv_s,
    const float max_eikonal_abs,
    const float occupancy_thresh,
    // const bool check_neighbours_density,
    //output
    // torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grid_tensor
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> grid_values_tensor,
    torch::PackedTensorAccessor32<bool,1,torch::RestrictPtrTraits> grid_occupancy_tensor
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_voxels){ //don't go out of bounds
        return;
    }

    //get the idx of the voxel corresponding to this point
    int idx_voxel=idx;
    //the index here is already in morton order since the points were generated in morton order 

    float voxel_size=grid_extent/nr_voxels_per_dim;
    float half_voxel_size=voxel_size/2.0;
    float half_cube_diagonal=sqrtf(3.0)*half_voxel_size;

    // int x_idx = morton3D_invert(idx>>0); 
	// int y_idx = morton3D_invert(idx>>1); //right bit shift divide by 2 and floors
	// int z_idx = morton3D_invert(idx>>2);

    // if(idx<=10){
    //     printf("idx is %d,  x,y,z is %d, %d, %d\n", idx,x_idx, y_idx, z_idx);
    // }
    
    // float old_density=grid_tensor[x_idx][y_idx][z_idx];
    float old_sdf=grid_values_tensor[idx_voxel];


    // old_density=old_density*decay; //decay
    float new_sdf=sdf_tensor[idx][0];


    //update
    // float updated_sdf=fmax(new_density, old_density);
    float updated_sdf=new_sdf;
    //float exponential mean
    // float alpha=0.3;
    // float updated_sdf= old_sdf + alpha*(new_sdf-old_sdf);
    // grid_tensor[x_idx][y_idx][z_idx]=updated_density;


    grid_values_tensor[idx_voxel]=updated_sdf;

    //check if the sdf can posibly be 0 or within the range that it would get a slight density
    //check the sdf that can be reached within this voxel
    //the max_eikonal loss says how much the distance deviates from a truly metric distance, so when the max eikonal is 2 that means that a distance of unit, actually has an sdf of 2. When the error of max_eikonal is almost zero, then the maximum we can move is just the half of the cube diagonal
    // float sdf_error_range=1.1*half_cube_diagonal*(1.0+max_eikonal_abs); //the sdf can be sdf+[-sdf_error_range, sdf_error_range]
    float sdf_error_range=1.3*half_cube_diagonal; //the sdf can be sdf+[-sdf_error_range, sdf_error_range]
    float minimum_sdf_possible_in_voxel=fabs(updated_sdf)-sdf_error_range;
    float capped_minimum_sdf_possible_in_voxel=clamp(minimum_sdf_possible_in_voxel,0.0, 1e10);
    //pass this sdf through the logistic function that neus uses and check what density it gets
    // float weight=sigmoid(capped_minimum_sdf_possible_in_voxel* inv_s);
    float weight=logistic_density_distribution(capped_minimum_sdf_possible_in_voxel, inv_s);
    // printf(" updated_sdf is %f capped_minimum_sdf_possible_in_voxel is %f inv_s is %f weight is %f \n",updated_sdf, capped_minimum_sdf_possible_in_voxel, inv_s, weight);


    grid_occupancy_tensor[idx_voxel]=weight>occupancy_thresh;

}

__global__ void 
update_with_sdf_random_sample_gpu(
    const int nr_points,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> sdf_tensor,
    const float grid_extent,
    const int nr_voxels_per_dim,
    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> point_indices,  
    const float inv_s,
    // const float max_eikonal_abs,
    const float occupancy_thresh,
    // const bool check_neighbours_density,
    //output
    // torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grid_tensor
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> grid_values_tensor,
    torch::PackedTensorAccessor32<bool,1,torch::RestrictPtrTraits> grid_occupancy_tensor
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_points){ //don't go out of bounds
        return;
    }

    //get the idx of the voxel corresponding to this point
    int idx_voxel=point_indices[idx];
    //the index here is already in morton order since the points were generated in morton order 

    float voxel_size=grid_extent/nr_voxels_per_dim;
    float half_voxel_size=voxel_size/2.0;
    float half_cube_diagonal=sqrtf(3.0)*half_voxel_size;

    // int x_idx = morton3D_invert(idx>>0); 
	// int y_idx = morton3D_invert(idx>>1); //right bit shift divide by 2 and floors
	// int z_idx = morton3D_invert(idx>>2);

    // if(idx<=10){
    //     printf("idx is %d,  x,y,z is %d, %d, %d\n", idx,x_idx, y_idx, z_idx);
    // }
    
    // float old_density=grid_tensor[x_idx][y_idx][z_idx];
    float old_sdf=grid_values_tensor[idx_voxel];


    // old_density=old_density*decay; //decay
    float new_sdf=sdf_tensor[idx][0];


    //update
    // float updated_sdf=fmax(new_density, old_density);
    float updated_sdf=new_sdf;
    //float exponential mean
    // float alpha=0.3;
    // float updated_sdf= old_sdf + alpha*(new_sdf-old_sdf);
    // grid_tensor[x_idx][y_idx][z_idx]=updated_density;


    grid_values_tensor[idx_voxel]=updated_sdf;

    //check if the sdf can posibly be 0 or within the range that it would get a slight density
    //check the sdf that can be reached within this voxel
    //the max_eikonal loss says how much the distance deviates from a truly metric distance, so when the max eikonal is 2 that means that a distance of unit, actually has an sdf of 2. When the error of max_eikonal is almost zero, then the maximum we can move is just the half of the cube diagonal
    // float sdf_error_range=1.1*half_cube_diagonal*(1.0+max_eikonal_abs); //the sdf can be sdf+[-sdf_error_range, sdf_error_range]
    float sdf_error_range=1.0*half_cube_diagonal; //the sdf can be sdf+[-sdf_error_range, sdf_error_range]
    float minimum_sdf_possible_in_voxel=fabs(updated_sdf)-sdf_error_range;
    float capped_minimum_sdf_possible_in_voxel=clamp(minimum_sdf_possible_in_voxel,0.0, 1e10);
    //pass this sdf through the logistic function that neus uses and check what density it gets
    // float weight=sigmoid(capped_minimum_sdf_possible_in_voxel* inv_s);
    float weight=logistic_density_distribution(capped_minimum_sdf_possible_in_voxel, inv_s);
    // printf(" updated_sdf is %f capped_minimum_sdf_possible_in_voxel is %f inv_s is %f weight is %f \n",updated_sdf, capped_minimum_sdf_possible_in_voxel, inv_s, weight);


    grid_occupancy_tensor[idx_voxel]=weight>occupancy_thresh;

}


__global__ void 
compute_samples_in_occupied_regions_gpu(
    const int nr_rays,
    const int nr_voxels_per_dim,
    const float grid_extent,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> grid_translation_tensor,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_origins,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_dirs,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_t_entry,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_t_exit,
    const torch::PackedTensorAccessor32<bool,1,torch::RestrictPtrTraits> grid_occupancy_tensor,
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
    // torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> max_nr_samples,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> cur_nr_samples
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_rays){ //don't go out of bounds
        return;
    }

    //load everything for this ray
    float3 grid_translation=make_float3(grid_translation_tensor[0], grid_translation_tensor[1], grid_translation_tensor[2]);
    float3 ray_origin=make_float3(ray_origins[idx][0], ray_origins[idx][1], ray_origins[idx][2]);
    float3 ray_dir=make_float3(ray_dirs[idx][0], ray_dirs[idx][1], ray_dirs[idx][2]);
    // float3 inverse_dir=1.0/(ray_dir+1e-20);
    float3 inverse_dir; //save division in case soem entries are zero
    if(fabs(ray_dir.x)<1e-16){ inverse_dir.x=0; } else {inverse_dir.x=1.0/ray_dir.x;};
    if(fabs(ray_dir.y)<1e-16){ inverse_dir.y=0; } else {inverse_dir.y=1.0/ray_dir.y;};
    if(fabs(ray_dir.z)<1e-16){ inverse_dir.z=0; } else {inverse_dir.z=1.0/ray_dir.z;};
    // printf("inv dir is %f,%f,%f  \n", inverse_dir.x, inverse_dir.y, inverse_dir.z);
    float t_start=ray_t_entry[idx][0];
    float t_exit=ray_t_exit[idx][0];
    

    bool debug=false;
    if (debug){
        //create some samples in the occupied regions for debug purposes
        float t=t_start; //cur_t
        int nr_steps=0;
        while (t<t_exit &&nr_steps<MAX_STEPS()) {
            // printf("t is  %f, t_exit is %f nr_steps is %d  \n", t, t_exit, nr_steps);
            float3 pos = ray_origin+t*ray_dir;
            //if we are in an occupied voxel, accumulate the distance that we traversed through occupied space 
            // printf("pos is %f,%f,%f \n", pos.x, pos.y, pos.z);
            int idx_voxel=pos_to_lin_idx(pos, nr_voxels_per_dim, grid_extent, grid_translation, false); //important that the get_center_of_voxel is false regarldess of what you have in the function for compute_grid_points
            // printf("idxvoxel is %d \n", idx_voxel);s
            if(idx_voxel>=nr_voxels_per_dim*nr_voxels_per_dim*nr_voxels_per_dim || idx_voxel<0){
                break;
            }
            // printf("occupancy is  is %d \n", grid_occupancy_tensor[idx_voxel]);
            if (grid_occupancy_tensor[idx_voxel]){ //if it' occupied, we advance the t just a bit and create samples
                samples_pos[nr_steps][0]=pos.x;
                samples_pos[nr_steps][1]=pos.y;
                samples_pos[nr_steps][2]=pos.z;
                t+=0.01;
                atomicAdd(&cur_nr_samples[0],1);
            }else{
                //we are an in an empty voxel, so we advance to the next one
                t+=distance_to_next_voxel(pos, ray_dir, inverse_dir, nr_voxels_per_dim);
                t+=1e-6; //tiny epsilon so we make sure that we are now in the next voxel when we sample from it
                // t=advance_to_next_voxel(t,pos, ray_dir, inverse_dir, nr_voxels_per_dim);
                // printf("advanced to next voxel with a t of %f \n", t);
            }
            // if (grid_occupied_at(pos, density_grid, mip)) {
            // t = advance_to_next_voxel(t, cone_angle, pos, ray_d_normalized, idir, res);

            nr_steps+=1;
        }
    }else{

        float eps=1e-6; // float32 only has 7 decimal digits precision

        //run once through the occupancy to check how much distance do we traverse through occupied space
        float t=t_start; //cur_t
        int nr_steps=0;
        float distance_through_occupied_space=0.0;
        while (t<t_exit &&nr_steps<MAX_STEPS()) {
            float3 pos = ray_origin+t*ray_dir;
            int idx_voxel=pos_to_lin_idx(pos, nr_voxels_per_dim, grid_extent, grid_translation, false); //important that the get_center_of_voxel is false regarldess of what you have in the function for compute_grid_points
            if(idx_voxel>=nr_voxels_per_dim*nr_voxels_per_dim*nr_voxels_per_dim || idx_voxel<0){
                break;
            }
            float dist_to_next=distance_to_next_voxel(pos, ray_dir, inverse_dir, nr_voxels_per_dim);
            t+=dist_to_next;
            t+=eps; //tiny epsilon so we make sure that we are now in the next voxel when we sample from it
            //if we are in an occupied voxel, accumulate the distance that we traversed through occupied space 
            if (grid_occupancy_tensor[idx_voxel]){
                distance_through_occupied_space+=dist_to_next;
                //if the got outside of the bounding box with this last distance, substract the different so we are back in the boounding primitive
                if( (t-eps)>t_exit){
                    distance_through_occupied_space-= (t-eps)-t_exit;
                }
            }
            nr_steps+=1;
        }


        //now we have a maximum distance that we traverse through occupied space. 
        //we also have a min distance between samples, so we can calculate how many samples we need
        int nr_samples_to_create=distance_through_occupied_space/min_dist_between_samples;

        //this nr of samples can be quite large if the occupied distance is big, so we clamp the nr of samples
        nr_samples_to_create=clamp(nr_samples_to_create, 0, max_nr_samples_per_ray);

        //recalculate the dist between samples given the new nr of samples
        float dist_between_samples=distance_through_occupied_space/nr_samples_to_create;


        //if we have samples to create we create them
        //we also don't create anything if we have only 1 sample. Only 1 sample creates lots of probles with the cdf which has only one value of 0 and therefore binary search runs forever
        if (nr_samples_to_create>1){
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


            t=t_start; //cur_t
            nr_steps=0;

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
                int idx_voxel=pos_to_lin_idx(pos, nr_voxels_per_dim, grid_extent, grid_translation, false); //important that the get_center_of_voxel is false 
                if(idx_voxel>=nr_voxels_per_dim*nr_voxels_per_dim*nr_voxels_per_dim || idx_voxel<0){
                    break;
                }
                //we also check that we don't create more samples because sometimes it can happen to create one or more less depending on floating point errors
                if (grid_occupancy_tensor[idx_voxel] && nr_samples_created<nr_samples_to_create){ //if it' occupied, we advance the t just a bit and create samples
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
                }else{
                    //we are an in an empty voxel, so we advance to the next one
                    float delta=distance_to_next_voxel(pos, ray_dir, inverse_dir, nr_voxels_per_dim);
                    if(jitter_samples){ //have to jitter here to since we may start a new voxel and don;t want to make samples directly at the beggining of the voxel
                        //we already advanced the rng so no need to do it here
                        delta=delta+dist_between_samples*rng.next_float();
                    }
                    t+=delta;
                    t+=eps; //tiny epsilon so we make sure that we are now in the next voxel when we sample from it
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



















        // //go again through the occupancy grid and create the samples, we assume we have enough space for them
        // //first do some bookeeping for this ray, like the nr of samples, so we know where to write in the packed tensor
        // int indx_start_sample=atomicAdd(&cur_nr_samples[0],nr_samples_to_create);
        // ray_start_end_idx[idx][0]=indx_start_sample;
        // ray_start_end_idx[idx][1]=indx_start_sample+nr_samples_to_create-1; //we substract 1 because if we create for example 2 samples we want the indices to be the start idx to be 0 and the end idx to be 1. So the start and end idx point directly at the first and last sampler

        // //if we are about to create more samples that we can actually store, just exit this thread
        // //however we still keep writing into ray_start_end_idx because we want to be able later to filter those rays out that actually have no samples and not process them or not volumetrically integrate them
        // if( (indx_start_sample+nr_samples_to_create)>max_nr_samples){
        //     return;
        // }

        // t=t_start; //cur_t
        // nr_steps=0;

        // //jutter just the begginign so the samples are all at the same distance from each other and therefore the dt is all the same
        // float rand_mov=0;
        // if(jitter_samples){
        //     rng.advance(idx); //since all the threads start with the same seed, we need to advance this thread so it gets other numbers different than the otehrs
        //     float rand=rng.next_float();
        //     t=t+dist_between_samples*rand;
        // }


        // if(nr_samples_to_create==0){
        //     ray_fixed_dt[idx][0]=0;
        // }else{
        //     ray_fixed_dt[idx][0]=dist_between_samples;
        // }


        // int nr_samples_created=0;
        // while (t<t_exit &&nr_steps<MAX_STEPS()) {
        //     t=clamp(t, t_start, t_exit);
        //     float3 pos = ray_origin+t*ray_dir;
        //     int idx_voxel=pos_to_lin_idx(pos, nr_voxels_per_dim, grid_extent, grid_translation, false); //important that the get_center_of_voxel is false 
        //     //we also check that we don't create more samples because sometimes it can happen to create one or more less depending on floating point errors
        //     if (grid_occupancy_tensor[idx_voxel] && nr_samples_created<nr_samples_to_create){ //if it' occupied, we advance the t just a bit and create samples
        //         //store positions
        //         samples_pos[indx_start_sample+nr_samples_created][0]=pos.x;
        //         samples_pos[indx_start_sample+nr_samples_created][1]=pos.y;
        //         samples_pos[indx_start_sample+nr_samples_created][2]=pos.z;
        //         //store dirs
        //         samples_dirs[indx_start_sample+nr_samples_created][0]=ray_dir.x;
        //         samples_dirs[indx_start_sample+nr_samples_created][1]=ray_dir.y;
        //         samples_dirs[indx_start_sample+nr_samples_created][2]=ray_dir.z;
        //         //store z
        //         samples_z[indx_start_sample+nr_samples_created][0]=t;
        //         samples_dt[indx_start_sample+nr_samples_created][0]=dist_between_samples;
        //         //go to next sample
        //         // float rand_mov=0;
        //         // if(jitter_samples){
        //         //     rng.advance(idx); //since all the threads start with the same seed, we need to advance this thread so it gets other numbers different than the otehrs
        //         //     float rand=rng.next_float();
        //         //     rand_mov=dist_between_samples*rand-dist_between_samples/2.0; //moves between [-half_dist, half_dist]
        //         // }
        //         // t+=dist_between_samples+rand_mov;
        //         t+=dist_between_samples;
        //         // t=clamp(t, t_start, t_exit);
        //         nr_samples_created+=1;
        //     }else{
        //         //we are an in an empty voxel, so we advance to the next one
        //         float delta=distance_to_next_voxel(pos, ray_dir, inverse_dir, nr_voxels_per_dim);
        //         if(jitter_samples){ //have to jitter here to since we may start a new voxel and don;t want to make samples directly at the beggining of the voxel
        //             //we already advanced the rng so no need to do it here
        //             float rand=rng.next_float();
        //             delta=delta+dist_between_samples*rand;
        //         }
        //         t+=delta;
        //         t+=eps; //tiny epsilon so we make sure that we are now in the next voxel when we sample from it
        //     }
        //     nr_steps+=1;
        // }
        // //better to store exactly the nr of samples we created
        // // ray_start_end_idx[idx][0]=indx_start_sample;
        // // ray_start_end_idx[idx][1]=indx_start_sample+nr_samples_created;

        // //if we still have some more samples to create, we just set them to zero
        // for (int i=nr_samples_created; i<nr_samples_to_create; i++){
        //     //store positions
        //     samples_pos[indx_start_sample+i][0]=0;
        //     samples_pos[indx_start_sample+i][1]=0;
        //     samples_pos[indx_start_sample+i][2]=0;
        //     //store dirs
        //     samples_dirs[indx_start_sample+i][0]=0;
        //     samples_dirs[indx_start_sample+i][1]=0;
        //     samples_dirs[indx_start_sample+i][2]=0;
        //     //store z
        //     samples_z[indx_start_sample+i][0]=-1; //just a sentinel value that we can easily detect in the volumetric rendering and discard these samples
        //     samples_dt[indx_start_sample+i][0]=0;
        // }


        //sanity check that we created as many rays we intended to
        // if(nr_samples_to_create!=nr_samples_created){
            // printf("wtf is happening, nr_samples_to_create is %d while nr_samples_created is %d \n",nr_samples_to_create, nr_samples_created );
        // }



    }

    // t=startt;
	// j=0;
	// while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && j < numsteps) {
	// 	float dt = calc_dt(t, cone_angle);
	// 	uint32_t mip = mip_from_dt(dt, pos);
	// 	if (density_grid_occupied_at(pos, density_grid, mip)) {
	// 		coords_out(j)->set_with_optional_extra_dims(warp_position(pos, aabb), warped_dir, warp_dt(dt), extra_dims, coords_out.stride_in_bytes);
	// 		++j;
	// 		t += dt;
	// 	} else {
	// 		uint32_t res = NERF_GRIDSIZE()>>mip;
	// 		t = advance_to_next_voxel(t, cone_angle, pos, ray_d_normalized, idir, res);
	// 	}
	// }
	// if (max_level_rand_training) {
	// 	max_level_ptr += base;
	// 	for (j = 0; j < numsteps; ++j) {
	// 		max_level_ptr[j] = max_level;
	// 	}
	// }

}



__global__ void 
compute_first_sample_start_of_occupied_regions_gpu(
    const int nr_rays,
    const int nr_voxels_per_dim,
    const float grid_extent,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> grid_translation_tensor,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_origins,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_dirs,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_t_entry,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_t_exit,
    const torch::PackedTensorAccessor32<bool,1,torch::RestrictPtrTraits> grid_occupancy_tensor,
    const int max_nr_samples,
    // pcg32 rng,
    // const bool jitter_samples,
    //output
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_pos,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_dirs,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_z,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_dt,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_fixed_dt,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> ray_start_end_idx,
    // torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> max_nr_samples,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> cur_nr_samples
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_rays){ //don't go out of bounds
        return;
    }

    //load everything for this ray
    float3 grid_translation=make_float3(grid_translation_tensor[0], grid_translation_tensor[1], grid_translation_tensor[2]);
    float3 ray_origin=make_float3(ray_origins[idx][0], ray_origins[idx][1], ray_origins[idx][2]);
    float3 ray_dir=make_float3(ray_dirs[idx][0], ray_dirs[idx][1], ray_dirs[idx][2]);
    // float3 inverse_dir=1.0/(ray_dir+1e-20);
    float3 inverse_dir; //save division in case soem entries are zero
    if(fabs(ray_dir.x)<1e-16){ inverse_dir.x=0; } else {inverse_dir.x=1.0/ray_dir.x;};
    if(fabs(ray_dir.y)<1e-16){ inverse_dir.y=0; } else {inverse_dir.y=1.0/ray_dir.y;};
    if(fabs(ray_dir.z)<1e-16){ inverse_dir.z=0; } else {inverse_dir.z=1.0/ray_dir.z;};
    // printf("inv dir is %f,%f,%f  \n", inverse_dir.x, inverse_dir.y, inverse_dir.z);
    float t_start=ray_t_entry[idx][0];
    float t_exit=ray_t_exit[idx][0];
    

    bool debug=false;
    

    float eps=1e-6; // float32 only has 7 decimal digits precision
    int nr_samples_to_create=0;
    int nr_samples_created=0;

    float t=t_start; //cur_t
    int nr_steps=0;
    while (t<t_exit &&nr_steps<MAX_STEPS()) {
        float3 pos = ray_origin+t*ray_dir;
        int idx_voxel=pos_to_lin_idx(pos, nr_voxels_per_dim, grid_extent, grid_translation, false); //important that the get_center_of_voxel is false regarldess of what you have in the function for compute_grid_points
        if(idx_voxel>=nr_voxels_per_dim*nr_voxels_per_dim*nr_voxels_per_dim || idx_voxel<0){
            break;
        }
        float dist_to_next=distance_to_next_voxel(pos, ray_dir, inverse_dir, nr_voxels_per_dim);
        t+=dist_to_next;
        t+=eps; //tiny epsilon so we make sure that we are now in the next voxel when we sample from it
        //if we are in an occupied voxel, accumulate the distance that we traversed through occupied space 
        if (grid_occupancy_tensor[idx_voxel]){
            //create one sample and return
            nr_samples_to_create=1;


            int indx_start_sample=atomicAdd(&cur_nr_samples[0],nr_samples_to_create);
            ray_start_end_idx[idx][0]=indx_start_sample;
            ray_start_end_idx[idx][1]=indx_start_sample+nr_samples_to_create; 
            ray_fixed_dt[idx][0]=0;
            if( (indx_start_sample+nr_samples_to_create)>max_nr_samples){
                return;
            }
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
            samples_dt[indx_start_sample+nr_samples_created][0]=0;

            nr_samples_created+=1;

            return;
            
        }
    }

    //this ray passes onyl though unocupied space
    //we set the ray quantities to 0 and there is nothing to set in the per_sample quantities because we have no samples
    if(nr_samples_created<=2){
        ray_fixed_dt[idx][0]=0;
        ray_start_end_idx[idx][0]=0;
        ray_start_end_idx[idx][1]=0;
    }
    


    

    

}


__global__ void 
advance_sample_to_next_occupied_voxel_gpu(
    const int nr_points,
    const int nr_voxels_per_dim,
    const float grid_extent,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> grid_translation_tensor,
    // const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_origins,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_dirs,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_pos,
    const torch::PackedTensorAccessor32<bool,1,torch::RestrictPtrTraits> grid_occupancy_tensor,
    // pcg32 rng,
    // const bool jitter_samples,
    //output
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> new_samples_pos,
    torch::PackedTensorAccessor32<bool,2,torch::RestrictPtrTraits> is_within_bounds
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_points){ //don't go out of bounds
        return;
    }

    //load everything for this ray
    float3 grid_translation=make_float3(grid_translation_tensor[0], grid_translation_tensor[1], grid_translation_tensor[2]);
    float3 ray_origin=make_float3(samples_pos[idx][0], samples_pos[idx][1], samples_pos[idx][2]);
    float3 ray_dir=make_float3(samples_dirs[idx][0], samples_dirs[idx][1], samples_dirs[idx][2]);
    // float3 cur_pos_start=make_float3(samples_pos[idx][0], samples_pos[idx][1], samples_pos[idx][2]);
    // float3 inverse_dir=1.0/(ray_dir+1e-20);
    float3 inverse_dir; //save division in case soem entries are zero
    if(fabs(ray_dir.x)<1e-16){ inverse_dir.x=0; } else {inverse_dir.x=1.0/ray_dir.x;};
    if(fabs(ray_dir.y)<1e-16){ inverse_dir.y=0; } else {inverse_dir.y=1.0/ray_dir.y;};
    if(fabs(ray_dir.z)<1e-16){ inverse_dir.z=0; } else {inverse_dir.z=1.0/ray_dir.z;};
    // printf("inv dir is %f,%f,%f  \n", inverse_dir.x, inverse_dir.y, inverse_dir.z);
    // float t_start=ray_t_entry[idx][0];
    // float t_exit=ray_t_exit[idx][0];
    

    // bool debug=false;
    

    float eps=1e-6; // float32 only has 7 decimal digits precision
    int nr_samples_to_create=0;
    int nr_samples_created=0;

    float t=0; //cur_t
    int nr_steps=0;
    bool within_bounds=true;
    // while (within_bounds &&nr_steps<MAX_STEPS()) {
    while (within_bounds &&nr_steps<nr_voxels_per_dim*sqrt(3) ) {
        float3 pos = ray_origin+t*ray_dir;
        int idx_voxel=pos_to_lin_idx(pos, nr_voxels_per_dim, grid_extent, grid_translation, false); //important that the get_center_of_voxel is false regarldess of what you have in the function for compute_grid_points
        // if(idx_voxel>=nr_voxels_per_dim*nr_voxels_per_dim*nr_voxels_per_dim || idx_voxel<0){
        //     break;
        // }
        bool out_of_bounds=idx_voxel>(nr_voxels_per_dim*nr_voxels_per_dim*nr_voxels_per_dim-1) || idx_voxel<0;
        if (out_of_bounds){
            within_bounds=false;
            new_samples_pos[idx][0]=pos.x;
            new_samples_pos[idx][1]=pos.y;
            new_samples_pos[idx][2]=pos.z;
            break;
        }else{
            //advance towards next voxel and check if it's occupied
            float dist_to_next=distance_to_next_voxel(pos, ray_dir, inverse_dir, nr_voxels_per_dim);
            t+=dist_to_next;
            t+=eps; //tiny epsilon so we make sure that we are now in the next voxel when we sample from it
            //if we are in an occupied voxel, store this new sample point and break
            if (grid_occupancy_tensor[idx_voxel]){
                new_samples_pos[idx][0]=pos.x;
                new_samples_pos[idx][1]=pos.y;
                new_samples_pos[idx][2]=pos.z;
                break;
            }
        }

        nr_steps++;

       
    }

    is_within_bounds[idx][0]=within_bounds;

   
    


}





__global__ void 
check_occupancy_gpu(
    const int nr_points,
    const int nr_voxels_per_dim,
    const float grid_extent,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> grid_translation_tensor,
    const torch::PackedTensorAccessor32<bool,1,torch::RestrictPtrTraits> grid_occupancy_tensor,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> points,
    //output
    torch::PackedTensorAccessor32<bool,2,torch::RestrictPtrTraits> occupancy_value
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_points){ //don't go out of bounds
        return;
    }


    float3 grid_translation=make_float3(grid_translation_tensor[0], grid_translation_tensor[1], grid_translation_tensor[2]);


    float3 pos;
    pos.x=points[idx][0];
    pos.y=points[idx][1];
    pos.z=points[idx][2];
    int idx_voxel=pos_to_lin_idx(pos, nr_voxels_per_dim, grid_extent, grid_translation, false); //important that the get_center_of_voxel is false regarldess of what you have in the function for compute_grid_points


    //when you extract a mesh using marching cubes you may have some faces that slightly outside of the occupancy grid 
    bool out_of_bounds=idx_voxel>(nr_voxels_per_dim*nr_voxels_per_dim*nr_voxels_per_dim-1) || idx_voxel<0;
    // if(out_of_bounds){
        // printf("idx_voxel_out_of_bounds %d , pos is %f,%f,%f \n", idx_voxel, pos.x, pos.y, pos.z);
    // }

    if(!out_of_bounds){
        bool occ=grid_occupancy_tensor[idx_voxel];
        occupancy_value[idx][0]=occ;
    }else{
        //we are out of bounds so we will set occupancy to 0
        occupancy_value[idx][0]=false;
    }

}






}//namespace occupancy grid








