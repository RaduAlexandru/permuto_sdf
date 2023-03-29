#include "permuto_sdf/RaySamplesPacked.cuh"


#include "UtilsPytorch.h" //contains torch so it has to be added BEFORE any other include because the other ones might include loguru which gets screwed up if torch was included before it

#include "permuto_sdf/RaySamplesPackedGPU.cuh"





template <typename T>
T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}



RaySamplesPacked::RaySamplesPacked(const int nr_rays, const int nr_samples_maximum){
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

int RaySamplesPacked::compute_exact_nr_samples(){
    // nr_samples_per_ray=ray_start_end_idx[:,1:2]-ray_start_end_idx[:,0:1] 
    // https://stackoverflow.com/a/60059664
    torch::Tensor ray_start_idx=ray_start_end_idx.slice(1, 0, 1); //dim, start, end
    torch::Tensor ray_end_idx=ray_start_end_idx.slice(1, 1, 2);
    torch::Tensor nr_samples_per_ray=ray_end_idx-ray_start_idx;
    
    int nr_samples_total=nr_samples_per_ray.sum().item<int>();

    return nr_samples_total;
}


RaySamplesPacked RaySamplesPacked::compact_to_valid_samples(){


    int exact_nr_samples=this->compute_exact_nr_samples();

    RaySamplesPacked compact_ray_samples_packed(m_nr_rays, exact_nr_samples);
    compact_ray_samples_packed.has_sdf=this->has_sdf;
    compact_ray_samples_packed.rays_have_equal_nr_of_samples=this->rays_have_equal_nr_of_samples;
    compact_ray_samples_packed.fixed_nr_of_samples_per_ray=this->fixed_nr_of_samples_per_ray;

    const dim3 blocks = { (unsigned int)div_round_up(m_nr_rays, BLOCK_SIZE), 1, 1 }; 


    RaySamplesPackedGPU::compact_to_valid_samples_gpu<<<blocks, BLOCK_SIZE>>>(
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

    bool has_sdf=false;


}

void RaySamplesPacked::set_sdf(const torch::Tensor& sdf){
    samples_sdf=sdf.view({-1,1});
    has_sdf=true;
}
void RaySamplesPacked::remove_sdf(){
    has_sdf=false;
}

    

