#pragma once

#include "torch/torch.h"
#include <ATen/cuda/CUDAEvent.h>


//when we compute rays given the occupancy we have different number of samples for each ray but we want to have all of the samples in a packed and contiguous way
class RaySamplesPacked{
public:
    RaySamplesPacked(const int nr_rays, const int nr_samples_maximum);


    //since we preallocate more samples than necessary, some of them might be empty. so we just get the valid ones here
    RaySamplesPacked compact_to_valid_samples(); //incurrs a CPU sync but can be alleviatied if it's called some time after the kernels that created the RaySamplesPacked has been launched
    void set_sdf(const torch::Tensor& sdf);
    void remove_sdf();
    // int compute_exact_nr_samples();
    int compute_exact_nr_samples_cpu(); //blocks and sync to cpu to get the integer for the exact nr of samples
    int compute_exact_nr_rays_valid_cpu();
    torch::Tensor compute_exact_nr_samples_gpu(); //gets the nr of examples as a gpu tensor does not block
    torch::Tensor compute_exact_nr_rays_valid_gpu();
    void initialize_with_one_sample_per_ray(const torch::Tensor one_sample_per_ray, const torch::Tensor dirs); //usefult for creating samples when doing sphere tracing in which we have only one sample per ray. And then we can pass around this raysamples packed even if its compacted or not

    //synchronization things in order to avoid doing a compute_exact_nr_samples right after creating the RaySamplesPacked which will cause a long sync to gpu
    void enqueue_getting_exact_nr_samples(); //queues the cudamemcpyasync for reading the nr of samples and makes an event on GPU to check when that call is finished. this should not block
    void enqueue_getting_exact_nr_rays_valid();
    int wait_and_get_exact_nr_samples(); //we queued the reading of the exact nr of samples quite some time ago so luckily the kernels would be done and this doesn't block
    int wait_and_get_exact_nr_rays_valid();
    RaySamplesPacked compact_given_exact_nr_samples(const int exact_nr_samples); //does not block because we have already the nr of samples

    torch::Tensor m_nr_samples_cpu_async_transfer;  //used to issue a transfer from gpu to cpu asynchronously
    torch::Tensor m_nr_rays_valid_cpu_async_transfer;  //used to issue a transfer from gpu to cpu asynchronously
    std::shared_ptr<at::cuda::CUDAEvent> m_event_copy_nr_samples; //todo: change this because this is so ugly
    std::shared_ptr<at::cuda::CUDAEvent> m_event_copy_nr_rays_valid; 
        

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

    bool is_compact; //if the samples are not compacted it means that ray_start_end_idx are not contiguous, so one ray may span samples from [0:30] and the next ray samples from [55:65], the range between [30:55] are invalid samples which should be discarded with compact_to_valid_samples()

    //if it has sdf it should also be combined properly with another raysamples packed when needed
    bool has_sdf;


    int m_nr_rays;

};
