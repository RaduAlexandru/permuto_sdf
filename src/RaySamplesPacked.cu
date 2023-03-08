#include "hash_sdf/RaySamplesPacked.cuh"


#include "UtilsPytorch.h" //contains torch so it has to be added BEFORE any other include because the other ones might include loguru which gets screwed up if torch was included before it

#include "hash_sdf/RaySamplesPackedGPU.cuh"





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

//since we preallocate more samples than necessary, some of them might be empty. so we just get the valid ones here
//cannot do it this way because some rays may have allocated more samples that what they actually created so the cur_nr_samples may be higher than the actual nr of samples which would calculated from ray_start_end_idx[:,1]-ray_start_end_idx[:,0]. This causes further problems when doing the colorcal because we don't know how many weights and biases we should repeat for each pixel 
// RaySamplesPacked RaySamplesPacked::compact_to_valid_samples(){
//     torch::cuda::synchronize();
//     int cur_nr_samples_cpu=cur_nr_samples.cpu().item<int>(); 
//     if(cur_nr_samples_cpu>max_nr_samples){
//         std::cout <<"We created more samples than the maximum nr of samples that we can store in RaySamplesPacked. Please increse the maximum nr of samples in the OccupancyGrid class. Cur nr samples is  " << cur_nr_samples_cpu << "while max nr samples is " << max_nr_samples << std::endl;
//     }
//     cur_nr_samples_cpu=std::min(cur_nr_samples_cpu,max_nr_samples);
//     int nr_rays=ray_start_end_idx.size(0);

//     //make a struct with only the valid samples
//     RaySamplesPacked ray_samples_packed_valid(nr_rays, cur_nr_samples_cpu);
    
//     ray_samples_packed_valid.samples_pos=samples_pos.slice(0, 0, cur_nr_samples_cpu);
//     ray_samples_packed_valid.samples_pos_4d=samples_pos_4d.slice(0, 0, cur_nr_samples_cpu);
//     ray_samples_packed_valid.samples_dirs=samples_dirs.slice(0, 0, cur_nr_samples_cpu);
//     ray_samples_packed_valid.samples_z=samples_z.slice(0, 0, cur_nr_samples_cpu);
//     ray_samples_packed_valid.samples_dt=samples_dt.slice(0, 0, cur_nr_samples_cpu);
//     ray_samples_packed_valid.samples_sdf=samples_sdf.slice(0, 0, cur_nr_samples_cpu);
//     ray_samples_packed_valid.ray_fixed_dt=ray_fixed_dt;
//     ray_samples_packed_valid.ray_start_end_idx=ray_start_end_idx;

//     return ray_samples_packed_valid;
// }

RaySamplesPacked RaySamplesPacked::compact_to_valid_samples(){


    int exact_nr_samples=this->compute_exact_nr_samples();

    RaySamplesPacked compact_ray_samples_packed(m_nr_rays, exact_nr_samples);

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

void RaySamplesPacked::set_sdf(const torch::Tensor& sdf){
    samples_sdf=sdf.view({-1,1});
    has_sdf=true;
}

    

