#pragma once


#include <torch/torch.h>
#include "permuto_sdf/helper_math.h"

//Add this header after we add all cuda stuff because we need the profiler to have cudaDeviceSyncronize defined
// #define ENABLE_CUDA_PROFILING 1
// #include "Profiler.h" 

//matrices
#include "permuto_sdf/mat3.h"
#include "permuto_sdf/mat4.h"





#define BLOCK_SIZE 256

namespace VolumeRenderingGPU{

__device__ float map_range_val(const float input_val, const float input_start, const float input_end, const float  output_start, const float  output_end){
    float input_clamped=max(input_start, min(input_end, input_val));
    return output_start + ((output_end - output_start) / (input_end - input_start)) * (input_clamped - input_start);
}


//we reuse this function a lot so we just refactor it
__device__ float get_start_end_ray_indices(int& nr_samples_per_ray, int& out_idx_start, int& out_idx_end, const int idx_ray, const bool rays_have_equal_nr_of_samples, const int fixed_nr_of_samples_per_ray, const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits>& ray_start_end_idx  ){ 

    //consider the idx start and end pointing directly at the start and end sample. That is an issue because when the point at the same sample we cannot distinguish if the ray has 1 sample or 0
    // out_idx_start=0;
    // out_idx_end=0;
    // if (rays_have_equal_nr_of_samples){
    //     out_idx_start=idx_ray*fixed_nr_of_samples_per_ray;
    //     out_idx_end=idx_ray*fixed_nr_of_samples_per_ray+fixed_nr_of_samples_per_ray-1; //we do a -1 because idx_end has to point towards the last sample that we integrate. So when fixed_nr_of_samples is 2, then idx_end points at 1
    // }else{
    //     out_idx_start=ray_start_end_idx[idx_ray][0];
    //     out_idx_end=ray_start_end_idx[idx_ray][1];
    // }
    // nr_samples_per_ray=out_idx_end-out_idx_start+1; //we add a +1 because we store indices towards first and start samples. So when we have 2 samples the indices would be 0 and 1;


    //the idx start points at the first sample, the idx_end points at the next sample after the last. 
    //this makes it easy to distinguish when a ray has 0 samples because the idx_start=idx_end
    out_idx_start=0;
    out_idx_end=0;
    if (rays_have_equal_nr_of_samples){
        out_idx_start=idx_ray*fixed_nr_of_samples_per_ray;
        out_idx_end=idx_ray*fixed_nr_of_samples_per_ray+fixed_nr_of_samples_per_ray;
    }else{
        out_idx_start=ray_start_end_idx[idx_ray][0];
        out_idx_end=ray_start_end_idx[idx_ray][1];
    }
    nr_samples_per_ray=out_idx_end-out_idx_start; 


    
}

__device__ float clamp_min(float x, float a ){
  return max(a, x);
}



__global__ void 
volume_render_nerf(
    const int nr_rays,
    const bool use_ray_t_exit,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_t_exit,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> rgb_samples,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> radiance_samples,
    const int max_nr_samples,
    // const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_pos,
    // const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_dirs,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_z,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_dt,
    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> ray_start_end_idx,
    const bool rays_have_equal_nr_of_samples,
    const int fixed_nr_of_samples_per_ray,
    //output
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> pred_rgb,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> pred_depth,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> bg_transmittance,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> weight_per_sample
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_rays){ //don't go out of bounds
        return;
    }

    //get the indexes of the start and end sample
    int idx_start=0;
    int idx_end=0;
    int nr_samples=0;
    get_start_end_ray_indices(nr_samples, idx_start, idx_end, idx, rays_have_equal_nr_of_samples, fixed_nr_of_samples_per_ray, ray_start_end_idx  );

    // if (idx_end>max_nr_samples){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    if (idx_end>max_nr_samples || nr_samples==0){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    // if (false){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
        pred_rgb[idx][0]=0;
        pred_rgb[idx][1]=0;
        pred_rgb[idx][2]=0;
        pred_depth[idx][0]=0;
        bg_transmittance[idx][0]=1.0;
    }else{
        //we actually integrate here similar to https://github.com/NVlabs/instant-ngp/blob/e1d33a42a4de0b24237685f2ebdc07bcef1ecae9/src/testbed_nerf.cu#L1314
        float T = 1.f;
        float EPSILON = 1e-4f;
        float3 rgb_ray = make_float3(0,0,0);

        // int nr_samples=idx_end-idx_start+1; //we add a +1 because we store indices towards first and start samples. So when we have 2 samples the indices would be 0 and 1;
        float depth_ray = 0.f;
        for (int i=0; i<nr_samples; i++) {
            if (T < EPSILON) {
                break;
            }

            float3 rgb;
            rgb.x=rgb_samples[idx_start+i][0];
            rgb.y=rgb_samples[idx_start+i][1];
            rgb.z=rgb_samples[idx_start+i][2];
            float cur_depth = samples_z[idx_start+i][0];
            float dt=samples_dt[idx_start+i][0];

            float density = radiance_samples[idx_start+i][0];


            const float alpha = 1.f - __expf(-density * dt);
            const float weight = alpha * T;
            rgb_ray += weight * rgb;
            depth_ray += weight * cur_depth;
            T *= (1.f - alpha);

            weight_per_sample[idx_start+i][0]=weight;

        }

        //finish
        pred_rgb[idx][0]=rgb_ray.x;
        pred_rgb[idx][1]=rgb_ray.y;
        pred_rgb[idx][2]=rgb_ray.z;
        pred_depth[idx][0]=depth_ray;
        bg_transmittance[idx][0]=T;

    }


    

}


__global__ void 
volume_render_nerf_backward(
    const int nr_rays,
    const bool use_ray_t_exit,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> grad_pred_rgb,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> grad_bg_transmittance,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> grad_weight_per_sample,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> pred_rgb,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_t_exit,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> bg_transmittance,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> rgb_samples,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> radiance_samples,
    const int max_nr_samples,
    // const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_pos,
    // const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_dirs,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_dt,
    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> ray_start_end_idx,
    const bool rays_have_equal_nr_of_samples,
    const int fixed_nr_of_samples_per_ray,
    //output
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> grad_rgb_samples,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> grad_radiance_samples
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_rays){ //don't go out of bounds
        return;
    }


    //in the forward pass the pred_rgb is a sumation of all the rgb_samples multiplied by some weights which are a function of density
    // pred_rgb= rgb_samples[0] * w(density[0]) + rgb_samples[1] * w(density[1])
    //from the upstream gradient we have dL/dpred_rgb
    //we need to output in grad_rgb_samples and grad_radiance_samples the dL/rgb_samples and dL/radiance_samples
    //dL/rgb_samples  --------------- 
    //dL/rgb_samples= dL/pred_rgb * dpred_rgb/rgb_samples
    //dL/rgb_samples= dL/pred_rgb * w
    //dL/rgb_samples[sample_id]= dL/pred_rgb * w[sample_id]
    //dL/radiance_samples  --------------- 
    //dL/radiance_samples= dL/pred_rgb * dpred_rgb/radiance_samples
    //dL/radiance_samples= dL/pred_rgb * rgb_samples[sample_id] * dw/density
    //dw/density----- 
    //in the forward pass the w= 1-exp(-density*dt)  * T
    //but T also depends on the previous densities, this kinda complicates stuf....


    //get the indexes of the start and end sample
    int idx_start=0;
    int idx_end=0;
    int nr_samples=0;
    get_start_end_ray_indices(nr_samples, idx_start, idx_end, idx, rays_have_equal_nr_of_samples, fixed_nr_of_samples_per_ray, ray_start_end_idx  );

    // if (idx_end>max_nr_samples){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    if (idx_end>max_nr_samples || nr_samples==0){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    // if (false){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
        // pred_rgb[idx][0]=0;
        // pred_rgb[idx][1]=0;
        // pred_rgb[idx][2]=0;
        //TODO write onyl zeros
    }else{
        //we actually integrate here similar to https://github.com/NVlabs/instant-ngp/blob/e1d33a42a4de0b24237685f2ebdc07bcef1ecae9/src/testbed_nerf.cu#L1314
        float T = 1.f;
        float EPSILON = 1e-4f;
        
        float3 grad_pred_rgb_ray;
        grad_pred_rgb_ray.x=grad_pred_rgb[idx][0];
        grad_pred_rgb_ray.y=grad_pred_rgb[idx][1];
        grad_pred_rgb_ray.z=grad_pred_rgb[idx][2];

        float grad_bg_transmittance_ray=grad_bg_transmittance[idx][0];
        float last_T=bg_transmittance[idx][0];

        float3 rgb_ray_fully_integrated;
        rgb_ray_fully_integrated.x=pred_rgb[idx][0];
        rgb_ray_fully_integrated.y=pred_rgb[idx][1];
        rgb_ray_fully_integrated.z=pred_rgb[idx][2];

        float3 rgb_ray_up_until_now = make_float3(0,0,0);

        // int nr_samples=idx_end-idx_start+1; //we add a +1 because we store indices towards first and start samples. So when we have 2 samples the indices would be 0 and 1;
        int nr_samples_processed=0;
        for (int i=0; i<nr_samples; i++) {
            if (T < EPSILON) {
                break;
            }

            float grad_w=grad_weight_per_sample[idx_start+i][0];
            if(grad_w!=0){
                printf(" Didn't implement backward pass for the weight per sample. The grad_w is %f \n", grad_w);
            }

            float3 rgb;
            rgb.x=rgb_samples[idx_start+i][0];
            rgb.y=rgb_samples[idx_start+i][1];
            rgb.z=rgb_samples[idx_start+i][2]; 
            float dt=samples_dt[idx_start+i][0];
            float density = radiance_samples[idx_start+i][0];


            const float alpha = 1.f - __expf(-density * dt);
            const float weight = alpha * T;
            rgb_ray_up_until_now += weight * rgb;

            //grad_rgb_samples
            grad_rgb_samples[idx_start+i][0]=grad_pred_rgb_ray.x*weight;
            grad_rgb_samples[idx_start+i][1]=grad_pred_rgb_ray.y*weight;
            grad_rgb_samples[idx_start+i][2]=grad_pred_rgb_ray.z*weight;

            T *= (1.f - alpha); //has to be before he grad_sample is calculated
            //grad density_samples
            // https://github.com/NVlabs/instant-ngp/blob/master/src/testbed_nerf.cu#L1314
            // we know the suffix of this ray compared to where we are up to. note the suffix depends on this step's alpha as suffix = (1-alpha)*(somecolor), so dsuffix/dalpha = -somecolor = -suffix/(1-alpha)

            //I made the derivations of this again in the meta notebook and indeed it's correct and now I understand how that comes. 
            //the main idea is that in the forward pass pred_rgb=c0*w0 + c1*w1 + c2*w2
            //dPredRGB/density0 = c0* dw0/density_0  + c1*dw1/density_0  +c2*dw2/density_0
            //if you unroll the partial derivatives you will see that in each term of the sum resembels somewhat something of form Cm*Wn *dt
            //so it's kinda close to what we already calcualted in the forward pass 
            //the sum of all colors from this one until the end of the ray is already known and is the suffix 
            //with a bit of reordering you can then have dPredRGB/density0 = c0*t1*dt - dt*suffix
            //which cna be reordered in what we have here
            float3 suffix = rgb_ray_fully_integrated - rgb_ray_up_until_now;
            float grad=0;
            grad+=grad_pred_rgb_ray.x *dt *  (T * rgb.x - suffix.x);
            grad+=grad_pred_rgb_ray.y *dt *  (T * rgb.y - suffix.y);
            grad+=grad_pred_rgb_ray.z *dt *  (T * rgb.z - suffix.z);
            //when we also combine with backgroud, we also get a gradient WRT to the last T
            grad+=grad_bg_transmittance_ray*(-dt*last_T);
            grad_radiance_samples[idx_start+i][0]=grad;


            nr_samples_processed+=1;

        }

        //if we broke early from the loop, just put zeros in the output grads

        

    }


    

}



__global__ void 
compute_dt_gpu(
    const int nr_rays,
    const bool use_ray_t_exit,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_t_exit,
    const int max_nr_samples,
    // const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_pos,
    // const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_dirs,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_z,
    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> ray_start_end_idx,
    const bool rays_have_equal_nr_of_samples,
    const int fixed_nr_of_samples_per_ray,
    //output
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> dt_samples
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_rays){ //don't go out of bounds
        return;
    }

    //get the indexes of the start and end sample
    int idx_start=0;
    int idx_end=0;
    int nr_samples=0;
    get_start_end_ray_indices(nr_samples, idx_start, idx_end, idx, rays_have_equal_nr_of_samples, fixed_nr_of_samples_per_ray, ray_start_end_idx  );


    // if (idx_end>max_nr_samples){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    if (idx_end>max_nr_samples || nr_samples==0){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    // if (false){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
        // dt_samples[idx][0]=0;
    }else{
       
        for (int i=0; i<nr_samples; i++) {
            float cur_depth = samples_z[idx_start+i][0];
            float next_depth;
            if (i<nr_samples-1){ //everything except the last sample
                next_depth=samples_z[idx_start+i+1][0];
            }else{ //the last sample doesnt have a next one to grab the depth so we get the depth from the intersection with the aabb
                if (use_ray_t_exit){
                    next_depth=ray_t_exit[idx][0];
                }else{
                    next_depth=1e10; //use a gigantic distance indicating that this sample almost models infinity
                }
            }
            float dt=next_depth-cur_depth;

            dt_samples[idx_start+i][0]=dt;
            

        }
        

    }


    

}



__global__ void 
cumprod_alpha2transmittance_gpu(
    const int nr_rays,
    const int max_nr_samples,
    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> ray_start_end_idx,
    const bool rays_have_equal_nr_of_samples,
    const int fixed_nr_of_samples_per_ray,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> alpha_samples,
    //output
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> transmittance_samples,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> bg_transmittance
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_rays){ //don't go out of bounds
        return;
    }

    //get the indexes of the start and end sample
    int idx_start=0;
    int idx_end=0;
    int nr_samples=0;
    get_start_end_ray_indices(nr_samples, idx_start, idx_end, idx, rays_have_equal_nr_of_samples, fixed_nr_of_samples_per_ray, ray_start_end_idx  );

    // if (idx_end>max_nr_samples){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    if (idx_end>max_nr_samples || nr_samples==0){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    // if (false){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    }else{

        float T = 1.f;
       
        for (int i=0; i<nr_samples; i++) {
           

            //cumprod
            transmittance_samples[idx_start+i][0]=T;
            if (i<nr_samples-1){ //don't compute the product with the last one, because we want the BG transmittance to correspond to the last T that we stored in transmittance samples
                float alpha=alpha_samples[idx_start+i][0];
                // T *= (1.f - alpha);
                T *= (alpha); //we assume the alpha is already 1-alpha
            }
            

        }

        bg_transmittance[idx][0]=T;
        

    }

}


__global__ void 
integrate_with_weights_gpu(
    const int nr_rays,
    const int max_nr_samples,
    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> ray_start_end_idx,
    const bool rays_have_equal_nr_of_samples,
    const int fixed_nr_of_samples_per_ray,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> rgb_samples,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> weights_samples,
    //output
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> pred_rgb_per_ray
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_rays){ //don't go out of bounds
        return;
    }

    //get the indexes of the start and end sample
    int idx_start=0;
    int idx_end=0;
    int nr_samples=0;
    get_start_end_ray_indices(nr_samples, idx_start, idx_end, idx, rays_have_equal_nr_of_samples, fixed_nr_of_samples_per_ray, ray_start_end_idx  );

    // if (idx_end>max_nr_samples){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    if (idx_end>max_nr_samples || nr_samples==0){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    // if (false){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    }else{

        float3 rgb_ray = make_float3(0,0,0);


        for (int i=0; i<nr_samples; i++) {
           

            //integrate
            float3 rgb;
            rgb.x=rgb_samples[idx_start+i][0];
            rgb.y=rgb_samples[idx_start+i][1];
            rgb.z=rgb_samples[idx_start+i][2];
            float weight=weights_samples[idx_start+i][0];

            rgb_ray += weight * rgb;
            

        }

        //finish
        pred_rgb_per_ray[idx][0]=rgb_ray.x;
        pred_rgb_per_ray[idx][1]=rgb_ray.y;
        pred_rgb_per_ray[idx][2]=rgb_ray.z;
        

    }

}



__device__ float sigmoid(float x){ 
	float res= 1.0/(1.0+exp(-x));
    return res;
}

__global__ void 
sdf2alpha_gpu(
    const int nr_rays,
    const int max_nr_samples,
    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> ray_start_end_idx,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_fixed_dt,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_dt,
    const bool rays_have_equal_nr_of_samples,
    const int fixed_nr_of_samples_per_ray,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> sdf_samples,
    float inv_s,
    // float inv_s_min,
    // float inv_s_max,
    const bool dynamic_inv_s,
    const float inv_s_multiplier,
    //output
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> alpha_samples
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_rays){ //don't go out of bounds
        return;
    }


    //preload some things
    float dt_uniform_samples=ray_fixed_dt[idx][0]; //just used to calculate the inv_s. this is noy really the dt we should use because that can change on a per-sample basis
    //inv_s_max=1024
    //inv_s_min=64
    //found these values by playing around with debug_occupancy_grid and modifying them in the gui
    if (dynamic_inv_s){
        float inv_s_max=1024;
        float inv_s_min=64;
        inv_s=map_range_val(dt_uniform_samples, 0.0001, 0.01, inv_s_max, inv_s_min  );
    }
    inv_s=inv_s*inv_s_multiplier;

    //get the indexes of the start and end sample
    int idx_start=0;
    int idx_end=0;
    int nr_samples=0;
    get_start_end_ray_indices(nr_samples, idx_start, idx_end, idx, rays_have_equal_nr_of_samples, fixed_nr_of_samples_per_ray, ray_start_end_idx  );

    if (idx_end>max_nr_samples || nr_samples==0){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    // if (false){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    }else{


        for (int i=0; i<nr_samples-1; i++) { //WE DO this loop up until the last sample, because we need to acces in this loop both cur and next sample
           
            float dt=samples_dt[idx_start+i][0];


            float prev_sdf=sdf_samples[idx_start+i][0];
            float next_sdf=sdf_samples[idx_start+i+1][0];
            float mid_sdf = (prev_sdf + next_sdf) * 0.5;
            float cos_val = (next_sdf - prev_sdf) / clamp_min(dt, 1e-6); //TODO change to clamped_min
            cos_val = clamp(cos_val,-1e3, 0.0) ;
            float prev_esti_sdf = mid_sdf - cos_val * dt * 0.5;
            float next_esti_sdf = mid_sdf + cos_val * dt * 0.5;
            float prev_cdf = sigmoid(prev_esti_sdf * inv_s);
            float next_cdf = sigmoid(next_esti_sdf * inv_s);
            float alpha = (prev_cdf - next_cdf + 1e-6) / (prev_cdf + 1e-6); //todo changed to clamped min

            alpha_samples[idx_start+i][0]=alpha;


        }

        

    }

}

template<int val_dim>
__global__ void 
sum_over_each_ray_gpu(
    const int nr_rays,
    const int max_nr_samples,
    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> ray_start_end_idx,
    const bool rays_have_equal_nr_of_samples,
    const int fixed_nr_of_samples_per_ray,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> sample_values,
    //output
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> values_sum,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> values_sum_stored_per_sample
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_rays){ //don't go out of bounds
        return;
    }



    //get the indexes of the start and end sample
    int idx_start=0;
    int idx_end=0;
    int nr_samples=0;
    get_start_end_ray_indices(nr_samples, idx_start, idx_end, idx, rays_have_equal_nr_of_samples, fixed_nr_of_samples_per_ray, ray_start_end_idx  );

    // if (idx_end>max_nr_samples){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    if (idx_end>max_nr_samples || nr_samples==0){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    // if (false){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    }else{


        float val_sum[val_dim]{0};
        for (int i=0; i<nr_samples; i++) { 
            
            for (int v=0; v<val_dim; v++) {  
                float val=sample_values[idx_start+i][v];
                val_sum[v]+=val;
            }
            

        }


        //finish, writ ethe sum for the whole ray
        for (int v=0; v<val_dim; v++) {  
            values_sum[idx][v]=val_sum[v];
        }

        //store also the sum for each sample
        for (int i=0; i<nr_samples; i++) { 
            for (int v=0; v<val_dim; v++) {  
                values_sum_stored_per_sample[idx_start+i][v]=val_sum[v];
            }
        }

        

    }

}


__global__ void 
cumsum_over_each_ray_gpu(
    const int nr_rays,
    const int max_nr_samples,
    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> ray_start_end_idx,
    const bool rays_have_equal_nr_of_samples,
    const int fixed_nr_of_samples_per_ray,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> sample_values,
    const bool inverse,
    //output
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> values_cumsum_stored_per_sample
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_rays){ //don't go out of bounds
        return;
    }



    //get the indexes of the start and end sample
    int idx_start=0;
    int idx_end=0;
    int nr_samples=0;
    get_start_end_ray_indices(nr_samples, idx_start, idx_end, idx, rays_have_equal_nr_of_samples, fixed_nr_of_samples_per_ray, ray_start_end_idx  );

    if (idx_end>max_nr_samples || nr_samples==0){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    // if (false){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    }else{


        float val_cumsum=0;
        for (int i=0; i<nr_samples; i++) { 
            

            float val=0;
            if(inverse){
                val=sample_values[idx_end-1-i][0];
            }else{
                val=sample_values[idx_start+i][0];
            }
            val_cumsum+=val;

            if(inverse){
                values_cumsum_stored_per_sample[idx_end-1-i][0]=val_cumsum; 
            }else{
                values_cumsum_stored_per_sample[idx_start+i][0]=val_cumsum; 
            }

            
            

        }


        

    }

}





__global__ void 
compute_cdf_gpu(
    const int nr_rays,
    const int max_nr_samples,
    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> ray_start_end_idx,
    const bool rays_have_equal_nr_of_samples,
    const int fixed_nr_of_samples_per_ray,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> sample_weights,
    //output
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> sample_cdf
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_rays){ //don't go out of bounds
        return;
    }



    //get the indexes of the start and end sample
    int idx_start=0;
    int idx_end=0;
    int nr_samples=0;
    get_start_end_ray_indices(nr_samples, idx_start, idx_end, idx, rays_have_equal_nr_of_samples, fixed_nr_of_samples_per_ray, ray_start_end_idx  );

    if (idx_end>max_nr_samples || nr_samples==0){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    // if (false){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    }else{


        float cdf=0;
        for (int i=0; i<nr_samples; i++) { 
           
            
            sample_cdf[idx_start+i][0]=cdf; //this is correct, we start by writign 0 and we ignore the last weight and we don't sum it. Neus modulea actually doesnt compute the weight for the last sample 
            float weight=sample_weights[idx_start+i][0];
            //sanity check that the last weight is 0 and the last cdf we write should be almost 1
            if(i==nr_samples-1){
                if( fabs(cdf-1.0)>1e-3 ){
                    printf("cdf we wrote in the last value is not 1.0, it is %f nr uniform samples is %d  \n",cdf, nr_samples);
                }
            }
            cdf+=weight;
            
            

        }



        

    }

}




//https://stackoverflow.com/a/21662870
__device__ 
int midpoint(int a, int b){
    return a + (b-a)/2;
}

//return the index of the first value in the sample_cdf that has a value higher then val
__device__
int binary_search(const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits>& sample_cdf, float val, int imin, int imax){
    int nr_iters=0;
    while (imax >= imin) {
        int imid = midpoint(imin, imax);
        float cdf = sample_cdf[imid][0];
        if(cdf>val){
            imax=imid;
        }else{
            imin=imid;
        }

        if( (imax-imin)==1 ){
            return imax;
        }


        nr_iters+=1;



    }


    return imax;
}



__global__ void 
importance_sample_gpu(
    const int nr_rays,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_origins,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_dirs,
    const int max_nr_samples,
    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> ray_start_end_idx,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_fixed_dt,
    const bool rays_have_equal_nr_of_samples,
    const int fixed_nr_of_samples_per_ray,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> samples_z,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> sample_cdf,
    const int nr_importance_samples,
    pcg32 rng,
    const bool jitter_samples,
    //output
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> out_samples_pos,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> out_samples_dirs,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> out_samples_z,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> out_ray_start_end_idx
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_rays){ //don't go out of bounds
        return;
    }



    //get the indexes of the start and end sample
    int idx_start=0;
    int idx_end=0;
    int nr_samples=0;
    get_start_end_ray_indices(nr_samples, idx_start, idx_end, idx, rays_have_equal_nr_of_samples, fixed_nr_of_samples_per_ray, ray_start_end_idx  );

    //get the indices where we should write the importance samples
    int imp_idx_start=0;
    int imp_idx_end=0;
    int imp_nr_samples=0;
    get_start_end_ray_indices(imp_nr_samples, imp_idx_start, imp_idx_end, idx, true, nr_importance_samples, out_ray_start_end_idx  );

    //preload some other stuff
    float3 ray_origin=make_float3(ray_origins[idx][0], ray_origins[idx][1], ray_origins[idx][2]);
    float3 ray_dir=make_float3(ray_dirs[idx][0], ray_dirs[idx][1], ray_dirs[idx][2]);
    float fixed_dt=ray_fixed_dt[idx][0];



    // if (idx_end>max_nr_samples){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    if (idx_end>max_nr_samples || nr_samples==0){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    // if (false){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples

        //some rays can have no sample sbecause they end up in empty space but since we are assuming we have a dense vector of nr_raysx16 of importance samples, we set the importance samples to 0
            for (int i=0; i<nr_importance_samples; i++) { 
            //write the DUMMY sample
            //pos
            out_samples_pos[imp_idx_start+i][0]=0;
            out_samples_pos[imp_idx_start+i][1]=0;
            out_samples_pos[imp_idx_start+i][2]=0;
            //dir
            out_samples_dirs[imp_idx_start+i][0]=0;
            out_samples_dirs[imp_idx_start+i][1]=0;
            out_samples_dirs[imp_idx_start+i][2]=0;
            //z
            out_samples_z[imp_idx_start+i][0]=-1;
        }

    }else{


        for (int i=0; i<nr_importance_samples; i++) {
            //get the distance between samples, in the [0,1] range
            //if we want for example to create 1 sample, then the uniform rand should be 0.5, so just in the middle of the [0,1] range
            //if we create 2 samples, the uniform rand for each would be 0.33 and 0.66
            //we need to get this distance in between them (in the case of 2 samples the dist would be 0.33)
            //we do it by imagining we are creating nr_importance_samples+2 values in the range[0,1]. The extremes would be exactly at 0 and 1 but the second sample is exactly the distance we want
            float dist_in_uniform_space=1.0/ (nr_importance_samples+1);
            float uniform_rand=dist_in_uniform_space+i*dist_in_uniform_space;
            if(jitter_samples){
                rng.advance(idx); //since all the threads start with the same seed, we need to advance this thread so it gets other numbers different than the otehrs
                float rand=rng.next_float();
                //move [-0.5/nr_importance_samples, + 0.5/nr_importance_samples]
                float mov=dist_in_uniform_space/2.0;
                uniform_rand+=map_range_val(rand, 0.0, 1.0,  -mov, +mov );
            }


            //don't make the uniform rand too close to 1.0 or 0.0 because due to numerical errors you might not find a cdf that is exactly 1
            uniform_rand=clamp(uniform_rand, 0.0+1e-6, 1.0-1e-5);

            //do importance sampling given the cdf, take inspiration from SIGGRAPH 2021: Global Illumination Based on Surfels the part about importance sampling of the brdf https://www.youtube.com/watch?v=h1ocYFrtsM4
            //binary search for the indices where the cdf crosses this value
            int imax=binary_search(sample_cdf, uniform_rand, idx_start, idx_end-1);
            int imin=max(imax-1, 0);
            float cdf_max=sample_cdf[imax][0];
            float cdf_min=sample_cdf[imin][0];
            if(cdf_min>uniform_rand || cdf_max<uniform_rand){
                //compute the maximum cdf and minimum
                float minimum_cdf=99999;
                float maximum_cdf=-99999;
                for (int d=0; d<nr_samples; d++) {
                    float cdf=sample_cdf[idx_start+d][0];
                    if (cdf<minimum_cdf) minimum_cdf=cdf;
                    if (cdf>maximum_cdf) maximum_cdf=cdf;
                }
                // printf("wtf uniform_rand %f, cdf_min %f, cdf_max %f, imax is %d MinAndMax over all samples %f,%f nrsamples %d \n", uniform_rand, cdf_min, cdf_max, imax, minimum_cdf, maximum_cdf, nr_samples);

            }

            //get eh z values of the imin and imax
            float z_max=samples_z[imax][0];
            float z_min=samples_z[imin][0];
            float z_imp=map_range_val(uniform_rand, cdf_min, cdf_max,  z_min, z_max );
            //since for the uniform samples we assume the same, dt we want the importance samples to be not further than fixed_dt from the samples
            //get the closest between z_max and z_min and make z_imp to be at most fixed_dt away from them
            float dist_to_zmin=z_imp-z_min; //some positive value
            float dist_to_zmax=z_max-z_imp;
            if(dist_to_zmin<dist_to_zmax){ //we are closest to the previous sample
                dist_to_zmin=min(dist_to_zmin, fixed_dt);
                z_imp=z_min+ dist_to_zmin;
            }else{ //we are closest to the next sample so we snap to it
                dist_to_zmax=min(dist_to_zmax, fixed_dt);
                z_imp=z_max - dist_to_zmax;
            }

            //create the new importance sample
            float3 imp_sample_pos=ray_origin+z_imp*ray_dir;

            //write the new sample
            //pos
            out_samples_pos[imp_idx_start+i][0]=imp_sample_pos.x;
            out_samples_pos[imp_idx_start+i][1]=imp_sample_pos.y;
            out_samples_pos[imp_idx_start+i][2]=imp_sample_pos.z;
            //dir
            out_samples_dirs[imp_idx_start+i][0]=ray_dir.x;
            out_samples_dirs[imp_idx_start+i][1]=ray_dir.y;
            out_samples_dirs[imp_idx_start+i][2]=ray_dir.z;
            //z
            out_samples_z[imp_idx_start+i][0]=z_imp;


        }


        



        

    }

}



__global__ void 
combine_uniform_samples_with_imp_gpu(
    const int nr_rays,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_origins,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_dirs,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_t_exit,
    //samples_packed
    const int uniform_max_nr_samples,
    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> uniform_ray_start_end_idx,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> uniform_ray_fixed_dt,
    const bool uniform_rays_have_equal_nr_of_samples,
    const int uniform_fixed_nr_of_samples_per_ray,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> uniform_samples_z,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> uniform_samples_sdf,
    const bool uniform_has_sdf,
    //samples_imp
    const int imp_max_nr_samples,
    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> imp_ray_start_end_idx,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> imp_ray_fixed_dt,
    const bool imp_rays_have_equal_nr_of_samples,
    const int imp_fixed_nr_of_samples_per_ray,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> imp_samples_z,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> imp_samples_sdf,
    const bool imp_has_sdf,
    //combined stuff just for sanity checking
    const int combined_max_nr_samples,
    //output
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> combined_samples_pos,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> combined_samples_dirs,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> combined_samples_z,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> combined_samples_dt,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> combined_samples_sdf,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> combined_ray_fixed_dt,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> combined_ray_start_end_idx,
    // torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> max_nr_samples,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> combined_cur_nr_samples
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_rays){ //don't go out of bounds
        return;
    }



    //get the indexes of the start and end sample
    int uniform_idx_start=0;
    int uniform_idx_end=0;
    int uniform_nr_samples=0;
    get_start_end_ray_indices(uniform_nr_samples, uniform_idx_start, uniform_idx_end, idx, uniform_rays_have_equal_nr_of_samples, uniform_fixed_nr_of_samples_per_ray, uniform_ray_start_end_idx  );

    //get the indices where we should read the importance samples
    int imp_idx_start=0;
    int imp_idx_end=0;
    int imp_nr_samples=0;
    get_start_end_ray_indices(imp_nr_samples, imp_idx_start, imp_idx_end, idx, true, imp_fixed_nr_of_samples_per_ray, imp_ray_start_end_idx  );

    //preload some other stuff
    float3 ray_origin=make_float3(ray_origins[idx][0], ray_origins[idx][1], ray_origins[idx][2]);
    float3 ray_dir=make_float3(ray_dirs[idx][0], ray_dirs[idx][1], ray_dirs[idx][2]);
    float fixed_dt=uniform_ray_fixed_dt[idx][0];

    //caulcate how many samples we need for this ray
    int combined_nr_samples=uniform_nr_samples+imp_nr_samples;

    //too low nr of samples
    if(uniform_nr_samples<=1){
        //we set the ray quantities to 0 and there is nothing to set in the per_sample quantities because we have no samples
        combined_ray_fixed_dt[idx][0]=0;
        combined_ray_start_end_idx[idx][0]=0;
        combined_ray_start_end_idx[idx][1]=0;
        return;
    }
    
    //allocate similar to how we do in the occupancy grid
    int combined_indx_start_sample=atomicAdd(&combined_cur_nr_samples[0],combined_nr_samples);
    combined_ray_start_end_idx[idx][0]=combined_indx_start_sample;
    combined_ray_start_end_idx[idx][1]=combined_indx_start_sample+combined_nr_samples; 

    if( (combined_indx_start_sample+combined_nr_samples)>combined_max_nr_samples){
        printf("this really shouldn't happen that we are writing more samples than the combined_max_nr_samples. How did that happen \n");
        return;
    }


    float t_exit=ray_t_exit[idx][0];



    combined_ray_fixed_dt[idx][0]=fixed_dt;

    //start writing all the samples
    int idx_cur_uniform=0;
    int idx_cur_imp=0;
    bool finished_reading_all_uniform=false;
    bool finished_reading_all_imp=false;
    for(int i=0; i<combined_nr_samples; i++){
        //we need to sort them in increasing order to z 
        //write alsoa  dt which is just the capped distance towards the next one
        float z_uniform=0;
        if (!finished_reading_all_uniform){
            z_uniform=uniform_samples_z[uniform_idx_start+idx_cur_uniform][0];
        }else{
            z_uniform=1e10; //gigantic value just to make sure we never choose thise one again
        }
        float z_imp=0;
        if (!finished_reading_all_imp){
            z_imp=imp_samples_z[imp_idx_start+idx_cur_imp][0];
        }else{
            z_imp=1e10; //gigantic value just to make sure we never choose the imp again
        }
        bool adding_uniform_sample = z_uniform < z_imp;

        
        if(adding_uniform_sample){
            //write the curent uniform sample
            float3 pos = ray_origin+z_uniform*ray_dir;
            //store positions
            combined_samples_pos[combined_indx_start_sample+i][0]=pos.x;
            combined_samples_pos[combined_indx_start_sample+i][1]=pos.y;
            combined_samples_pos[combined_indx_start_sample+i][2]=pos.z;
            //store dirs
            combined_samples_dirs[combined_indx_start_sample+i][0]=ray_dir.x;
            combined_samples_dirs[combined_indx_start_sample+i][1]=ray_dir.y;
            combined_samples_dirs[combined_indx_start_sample+i][2]=ray_dir.z;
            //store z
            combined_samples_z[combined_indx_start_sample+i][0]=z_uniform;
            //store sdf if needed
            if (uniform_has_sdf){
                combined_samples_sdf[combined_indx_start_sample+i][0]=uniform_samples_sdf[uniform_idx_start+idx_cur_uniform][0];
            }
            //DT write later, in another loop after we write all the other samples
            // samples_dt[indx_start_sample+nr_samples_created][0]=dist_between_samples;
            idx_cur_uniform+=1;
            if (idx_cur_uniform>=uniform_nr_samples){
                finished_reading_all_uniform=true;
            }
        }else{
            // printf("Adding z_imp %f \n", z_imp);
            //write the current importance sample
            float3 pos = ray_origin+z_imp*ray_dir;
            //store positions
            combined_samples_pos[combined_indx_start_sample+i][0]=pos.x;
            combined_samples_pos[combined_indx_start_sample+i][1]=pos.y;
            combined_samples_pos[combined_indx_start_sample+i][2]=pos.z;
            //store dirs
            combined_samples_dirs[combined_indx_start_sample+i][0]=ray_dir.x;
            combined_samples_dirs[combined_indx_start_sample+i][1]=ray_dir.y;
            combined_samples_dirs[combined_indx_start_sample+i][2]=ray_dir.z;
            //store z
            combined_samples_z[combined_indx_start_sample+i][0]=z_imp;
            //store sdf if needed
            if (imp_has_sdf){
                combined_samples_sdf[combined_indx_start_sample+i][0]=imp_samples_sdf[imp_idx_start+idx_cur_imp][0];
            }
            //DT write later, in another loop after we write all the other samples
            idx_cur_imp+=1;
            if (idx_cur_imp>=imp_nr_samples){
                finished_reading_all_imp=true;
            }
        }

    }


    //now, that we have all the samples, we can also write dt since dt is the distance from cur to next
    for(int i=0; i<combined_nr_samples-1; i++){
        float cur_z=combined_samples_z[combined_indx_start_sample+i][0];
        float next_z=combined_samples_z[combined_indx_start_sample+i+1][0];
        float dt=next_z-cur_z;
        dt=min(dt, fixed_dt);
        combined_samples_dt[combined_indx_start_sample+i][0]=dt;
    }
    //last combined sample
    float last_sample_z=combined_samples_z[combined_indx_start_sample+combined_nr_samples-1][0];
    float remaining_dist_until_border=t_exit-last_sample_z;
    combined_samples_dt[combined_indx_start_sample+combined_nr_samples-1][0]=clamp(remaining_dist_until_border, 0.0, fixed_dt);
    // combined_samples_dt[combined_indx_start_sample+combined_nr_samples-1][0]=fixed_dt;


}



__global__ void 
cumprod_alpha2transmittance_backward_gpu(
    const int nr_rays,
    const int max_nr_samples,
    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> ray_start_end_idx,
    const bool rays_have_equal_nr_of_samples,
    const int fixed_nr_of_samples_per_ray,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> grad_transmittance,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> grad_bg_transmittance,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> alpha,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> transmittance,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> bg_transmittance,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> cumsumLV,
    //output
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> grad_alpha_samples
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_rays){ //don't go out of bounds
        return;
    }

    //get the indexes of the start and end sample
    int idx_start=0;
    int idx_end=0;
    int nr_samples=0;
    get_start_end_ray_indices(nr_samples, idx_start, idx_end, idx, rays_have_equal_nr_of_samples, fixed_nr_of_samples_per_ray, ray_start_end_idx  );

    if (idx_end>max_nr_samples || nr_samples==0){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    // if (false){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    }else{

        float bg_transmittance_cur=bg_transmittance[idx][0]; //is the multiplication of all the alphas until now 
        float grad_bg_transmittance_cur=grad_bg_transmittance[idx][0]; //is the multiplication of all the alphas until now 

        for (int i=0; i<nr_samples; i++) { 

            float grad_alpha=0;
            float alpha_cur=alpha[idx_start+i][0];
            if(i<nr_samples-1){ //this skips the last sample which doesn't participate in the cumprod and therefore gets no gradient from this
                float cumsum_next=cumsumLV[idx_start+i+1][0];
                grad_alpha=cumsum_next/ clamp_min(alpha_cur,1e-6);

                // //accumulate alpha gradient due to the bg_alpha
                // if(fabs(grad_bg_transmittance_cur)>1e-6 && fabs(alpha_cur)>1e-6){
                    // grad_alpha+=grad_bg_transmittance_cur * bg_transmittance_cur / (alpha_cur+1e-6);
                    // grad_alpha+=grad_bg_transmittance_cur * bg_transmittance_cur / alpha_cur;
                // }
                grad_alpha+=grad_bg_transmittance_cur * bg_transmittance_cur / clamp_min(alpha_cur,1e-6);

            }


            // //accumulate alpha gradient due to the bg_alpha
            // if(fabs(grad_bg_transmittance_cur)>1e-6 && fabs(alpha_cur)>1e-6){
                // grad_alpha+=grad_bg_transmittance_cur * bg_transmittance_cur / alpha_cur;
            // }

            grad_alpha_samples[idx_start+i][0]=grad_alpha; //the last sample gets a grad_alpha of zero but this is fine because according to sdf2alpha the last sample has either way an alpha of zero

            

            
            

        }

    }

}


__global__ void 
integrate_with_weights_backward_gpu(
    const int nr_rays,
    const int max_nr_samples,
    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> ray_start_end_idx,
    const bool rays_have_equal_nr_of_samples,
    const int fixed_nr_of_samples_per_ray,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> grad_pred_rgb,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> rgb_samples,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> weights_samples,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> pred_rgb_per_ray,
    //output
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> grad_rgb_samples,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> grad_weights_samples
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_rays){ //don't go out of bounds
        return;
    }

    //get the indexes of the start and end sample
    int idx_start=0;
    int idx_end=0;
    int nr_samples=0;
    get_start_end_ray_indices(nr_samples, idx_start, idx_end, idx, rays_have_equal_nr_of_samples, fixed_nr_of_samples_per_ray, ray_start_end_idx  );

    if (idx_end>max_nr_samples || nr_samples==0){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    // if (false){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    }else{

        // float3 rgb_ray = make_float3(0,0,0);

        float3 grad_pred_rgb_cur_ray=make_float3(grad_pred_rgb[idx][0], grad_pred_rgb[idx][1], grad_pred_rgb[idx][2]  );


        for (int i=0; i<nr_samples; i++) {

            float3 rgb_cur_sample=make_float3(rgb_samples[idx_start+i][0], rgb_samples[idx_start+i][1], rgb_samples[idx_start+i][1] );
            float weight_cur_sample=weights_samples[idx_start+i][0];


            grad_rgb_samples[idx_start+i][0]=grad_pred_rgb_cur_ray.x * weight_cur_sample;
            grad_rgb_samples[idx_start+i][1]=grad_pred_rgb_cur_ray.y * weight_cur_sample;
            grad_rgb_samples[idx_start+i][2]=grad_pred_rgb_cur_ray.z * weight_cur_sample;

            grad_weights_samples[idx_start+i][0]=
                grad_pred_rgb_cur_ray.x * rgb_cur_sample.x + 
                grad_pred_rgb_cur_ray.y * rgb_cur_sample.y +
                grad_pred_rgb_cur_ray.z * rgb_cur_sample.z;

            

        }

      
        

    }

}

template<int val_dim>
__global__ void 
sum_over_each_ray_backward_gpu(
    const int nr_rays,
    const int max_nr_samples,
    const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> ray_start_end_idx,
    const bool rays_have_equal_nr_of_samples,
    const int fixed_nr_of_samples_per_ray,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> grad_values_sum_per_ray,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> grad_values_sum_per_sample,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> sample_values,
    //output
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> grad_sample_values
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_rays){ //don't go out of bounds
        return;
    }



    //get the indexes of the start and end sample
    int idx_start=0;
    int idx_end=0;
    int nr_samples=0;
    get_start_end_ray_indices(nr_samples, idx_start, idx_end, idx, rays_have_equal_nr_of_samples, fixed_nr_of_samples_per_ray, ray_start_end_idx  );

    if (idx_end>max_nr_samples || nr_samples==0){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    // if (false){ //this batch of samples would have ended up outside of the maximum allocation of samples but we didn't actually write any samples
    }else{


        // float val_sum=0;
        // float grad_sum_per_ray=grad_values_sum_per_ray[idx][0];
        //load the grad for the whole ray
        float grad_sum_per_ray[val_dim]{0};
        for (int v=0; v<val_dim; v++) { 
            grad_sum_per_ray[v]=grad_values_sum_per_ray[idx][v];
        }



        for (int i=0; i<nr_samples; i++) { 

            for (int v=0; v<val_dim; v++) { 
                float grad_per_sample=grad_values_sum_per_sample[idx_start+i][v];
                grad_sample_values[idx_start+i][v]= grad_sum_per_ray[v]+grad_per_sample;
            }
            
            

        }


    }

}


} //namespace VolumeRenderingGPU
