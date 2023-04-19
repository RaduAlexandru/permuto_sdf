#include "permuto_sdf/RaySamplesPacked.cuh"


#include "UtilsPytorch.h" //contains torch so it has to be added BEFORE any other include because the other ones might include loguru which gets screwed up if torch was included before it

#include "permuto_sdf/RaySamplesPackedGPU.cuh"





template <typename T>
T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}



RaySamplesPacked::RaySamplesPacked(const int nr_rays, const int nr_samples_maximum):
    is_compact(false)
    {
    m_nr_rays=nr_rays;
    max_nr_samples=nr_samples_maximum;

    cur_nr_samples=torch::empty({ 1 },  torch::dtype(torch::kInt32).device(torch::kCUDA, 0)  );
    cur_nr_samples.fill_(0);

    //preallocate the samples tensor
    samples_pos=torch::empty({ nr_samples_maximum,3 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    samples_pos_4d=torch::empty({ nr_samples_maximum,4 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    samples_dirs=torch::empty({ nr_samples_maximum,3 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    samples_z=torch::empty({ nr_samples_maximum,1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    samples_dt=torch::empty({ nr_samples_maximum,1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    samples_sdf=torch::empty({ nr_samples_maximum,1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    ray_fixed_dt=torch::empty({ nr_rays,1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    ray_start_end_idx=torch::empty({ nr_rays,2 },  torch::dtype(torch::kInt32).device(torch::kCUDA, 0)  );
    // ray_minimum_possible_sdf=torch::empty({ nr_rays,1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );

    rays_have_equal_nr_of_samples=false;
    fixed_nr_of_samples_per_ray=0;

    has_sdf=false;

}

torch::Tensor RaySamplesPacked::compute_exact_nr_samples_gpu(){
    // nr_samples_per_ray=ray_start_end_idx[:,1:2]-ray_start_end_idx[:,0:1] 
    // https://stackoverflow.com/a/60059664
    torch::Tensor ray_start_idx=ray_start_end_idx.slice(1, 0, 1); //dim, start, end
    torch::Tensor ray_end_idx=ray_start_end_idx.slice(1, 1, 2);
    torch::Tensor nr_samples_per_ray=ray_end_idx-ray_start_idx;
    
    // torch::Tensor nr_samples_total=nr_samples_per_ray.sum().item<int>();
    torch::Tensor nr_samples_total_gpu=nr_samples_per_ray.sum();

    return nr_samples_total_gpu;
}
int RaySamplesPacked::compute_exact_nr_samples_cpu(){
    return compute_exact_nr_samples_gpu().item<int>();
}

torch::Tensor RaySamplesPacked::compute_exact_nr_rays_valid_gpu(){
    // nr_samples_per_ray=ray_start_end_idx[:,1:2]-ray_start_end_idx[:,0:1] 
    // https://stackoverflow.com/a/60059664
    torch::Tensor ray_start_idx=ray_start_end_idx.slice(1, 0, 1); //dim, start, end
    torch::Tensor ray_end_idx=ray_start_end_idx.slice(1, 1, 2);
    torch::Tensor nr_samples_per_ray=ray_end_idx-ray_start_idx;
   
    torch::Tensor is_ray_valid=nr_samples_per_ray.ge(1);
    torch::Tensor nr_rays_valid_gpu=is_ray_valid.sum();

    return nr_rays_valid_gpu;
}
int RaySamplesPacked::compute_exact_nr_rays_valid_cpu(){
    return compute_exact_nr_rays_valid_gpu().item<int>();
}


RaySamplesPacked RaySamplesPacked::compact_to_valid_samples(){
    int exact_nr_samples=this->compute_exact_nr_samples_cpu();
    return compact_given_exact_nr_samples(exact_nr_samples);
}


void RaySamplesPacked::enqueue_getting_exact_nr_samples(){
    torch::Tensor nr_samples_gpu=compute_exact_nr_samples_gpu();

    bool non_blocking = true;
    bool copy_flag = false;
    m_nr_samples_cpu_async_transfer = nr_samples_gpu.to(torch::kCPU, non_blocking, copy_flag);
    //create event to check when this transfer is finsished 
    m_event_copy_nr_samples = std::make_unique<at::cuda::CUDAEvent>();
    m_event_copy_nr_samples->record(at::cuda::getCurrentCUDAStream());
    
}
int RaySamplesPacked::wait_and_get_exact_nr_samples(){
    CHECK(m_nr_samples_cpu_async_transfer.is_pinned()) << "The CPU tensor has to be pinned in order to enable async copying";
    //wait for event. After waiting, we will be sure that the transfer to m_nr_samples_cpu_async_transfer is finished
    m_event_copy_nr_samples->synchronize();
    return m_nr_samples_cpu_async_transfer.item<int>(); //is already on CPU so this will not block
}



void RaySamplesPacked::enqueue_getting_exact_nr_rays_valid(){
    torch::Tensor nr_rays_valid_gpu=compute_exact_nr_rays_valid_gpu();

    bool non_blocking = true;
    bool copy_flag = false;
    m_nr_rays_valid_cpu_async_transfer = nr_rays_valid_gpu.to(torch::kCPU, non_blocking, copy_flag);
    //create event to check when this transfer is finsished 
    m_event_copy_nr_rays_valid = std::make_unique<at::cuda::CUDAEvent>();
    m_event_copy_nr_rays_valid->record(at::cuda::getCurrentCUDAStream());
}

int RaySamplesPacked::wait_and_get_exact_nr_rays_valid(){
    CHECK(m_nr_rays_valid_cpu_async_transfer.is_pinned()) << "The CPU tensor has to be pinned in order to enable async copying";
    //wait for event. After waiting, we will be sure that the transfer to m_nr_samples_cpu_async_transfer is finished
    m_event_copy_nr_rays_valid->synchronize();
    return m_nr_rays_valid_cpu_async_transfer.item<int>(); //is already on CPU so this will not block
}




RaySamplesPacked RaySamplesPacked::compact_given_exact_nr_samples(const int exact_nr_samples){

    RaySamplesPacked compact_ray_samples_packed(m_nr_rays, exact_nr_samples);
    compact_ray_samples_packed.has_sdf=this->has_sdf;
    compact_ray_samples_packed.rays_have_equal_nr_of_samples=this->rays_have_equal_nr_of_samples;
    compact_ray_samples_packed.fixed_nr_of_samples_per_ray=this->fixed_nr_of_samples_per_ray;
    compact_ray_samples_packed.is_compact=true;

    const dim3 blocks = { (unsigned int)div_round_up(m_nr_rays, BLOCK_SIZE), 1, 1 }; 


    RaySamplesPackedGPU::compact_to_valid_samples_gpu<<<blocks, BLOCK_SIZE, 0, at::cuda::getDefaultCUDAStream()>>>(
                m_nr_rays,
                this->samples_pos.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                this->samples_pos_4d.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                this->samples_dirs.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                this->samples_z.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                this->samples_dt.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                this->samples_sdf.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                this->ray_fixed_dt.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                this->ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                //output
                compact_ray_samples_packed.samples_pos.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                compact_ray_samples_packed.samples_pos_4d.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                compact_ray_samples_packed.samples_dirs.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                compact_ray_samples_packed.samples_z.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                compact_ray_samples_packed.samples_dt.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                compact_ray_samples_packed.samples_sdf.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                compact_ray_samples_packed.ray_fixed_dt.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                compact_ray_samples_packed.ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                compact_ray_samples_packed.cur_nr_samples.packed_accessor32<int,1,torch::RestrictPtrTraits>()
            );



    return compact_ray_samples_packed;
}

void RaySamplesPacked::initialize_with_one_sample_per_ray(const torch::Tensor one_sample_per_ray, const torch::Tensor dirs){

    int nr_rays=one_sample_per_ray.size(0);

    samples_pos=one_sample_per_ray;
    // samples_pos_4d
    samples_dirs=dirs;
    samples_z=torch::zeros({ nr_rays,1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    samples_dt=torch::zeros({ nr_rays,1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    // samples_sdf
    ray_fixed_dt=torch::zeros({ nr_rays,1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );

    torch::Tensor idx_start=torch::arange(nr_rays).view({-1,1});
    torch::Tensor idx_end=idx_start+1;
    ray_start_end_idx=torch::cat({idx_start, idx_end},1);

    int max_nr_samples=nr_rays;
    cur_nr_samples.fill_(nr_rays);

    rays_have_equal_nr_of_samples=true;
    fixed_nr_of_samples_per_ray=1;

    has_sdf=false;
    is_compact=true;


}

void RaySamplesPacked::set_sdf(const torch::Tensor& sdf){
    samples_sdf=sdf.view({-1,1});
    has_sdf=true;
}
void RaySamplesPacked::remove_sdf(){
    has_sdf=false;
}

    

