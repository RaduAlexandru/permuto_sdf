#pragma once

#include "torch/torch.h"


//when we compute rays given the occupancy we have different number of samples for each ray but we want to have all of the samples in a packed and contiguous way
class RaySamplesPacked{
public:
    RaySamplesPacked(const int nr_rays, const int nr_samples_maximum);


    //since we preallocate more samples than necessary, some of them might be empty. so we just get the valid ones here
    RaySamplesPacked compact_to_valid_samples();
    void set_sdf(const torch::Tensor& sdf);
    void remove_sdf();
    int compute_exact_nr_samples();
    void initialize_with_one_sample_per_ray(const torch::Tensor one_sample_per_ray, const torch::Tensor dirs); //usefult for creating samples when doing sphere tracing in which we have only one sample per ray. And then we can pass around this raysamples packed even if its compacted or not
        

    //we keep the atributes in different tensors because it makes it easier to encode the directions later by just applying SH over the samples_dirs
    torch::Tensor samples_pos; //nr_samples x 3
    torch::Tensor samples_pos_4d; //nr_samples x 4 // for background modelling with nerf++
    torch::Tensor samples_dirs; //nr_samples x 3
    torch::Tensor samples_z; //nr_samples x 1
    torch::Tensor samples_dt; //nr_samples x 1 //the dt between each samples, we assume that all samples in the same ray are equidistance, even if they skip from one voxel to another
    torch::Tensor samples_sdf; //nr_samples x 1 //DO NOT write directly to this vector, rather use the set_sdf function
    torch::Tensor ray_fixed_dt; //since all the samples on the ray have the same dt when we create it with the occupancy grid, we can also store here the dt for that ray
    torch::Tensor ray_start_end_idx; // nr_rays x 2 for each ray, we store the idx of the first sample and the end sample that indexes into the sample_ tensors

    //we need a very large pool of samples because we are not quite sure how many we will create on every iteration
    // torch::Tensor max_nr_samples;
    int max_nr_samples; //maximum nr of samples that can be stored in this container. When cur_nr_samples approaches max_nr_samples, we should raise a warning
    torch::Tensor cur_nr_samples; //This will only be a conservative estimate of the nr of samples as it will always >= compute_exact_nr_samples(). This is because when samples are created in the occupancy grid the actual samples created per ray and indicated by the ray_start_end_idx may be smaller for each ray than what we allocate with the atomic add on cur_nr_samples

    //if the nr of samples per ray is always the same, then we don't need to read from ray_start_end_idx
    bool rays_have_equal_nr_of_samples;
    int fixed_nr_of_samples_per_ray;

    //if it has sdf it should also be combined properly with another raysamples packed when needed
    bool has_sdf;


    int m_nr_rays;

};
