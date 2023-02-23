#include "hash_sdf/RaySamplesPacked.cuh"


RaySamplesPacked::RaySamplesPacked(const int nr_rays, const int nr_samples_maximum){
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

//since we preallocate more samples than necessary, some of them might be empty. so we just get the valid ones here
RaySamplesPacked RaySamplesPacked::get_valid_samples(){
    torch::cuda::synchronize();
    int cur_nr_samples_cpu=cur_nr_samples.cpu().item<int>(); 
    if(cur_nr_samples_cpu>max_nr_samples){
        std::cout <<"We created more samples than the maximum nr of samples that we can store in RaySamplesPacked. Please increse the maximum nr of samples in the OccupancyGrid class. Cur nr samples is  " << cur_nr_samples_cpu << "while max nr samples is " << max_nr_samples << std::endl;
    }
    cur_nr_samples_cpu=std::min(cur_nr_samples_cpu,max_nr_samples);
    int nr_rays=ray_start_end_idx.size(0);

    //make a struct with only the valid samples
    RaySamplesPacked ray_samples_packed_valid(nr_rays, cur_nr_samples_cpu);
    
    ray_samples_packed_valid.samples_pos=samples_pos.slice(0, 0, cur_nr_samples_cpu);
    ray_samples_packed_valid.samples_pos_4d=samples_pos_4d.slice(0, 0, cur_nr_samples_cpu);
    ray_samples_packed_valid.samples_dirs=samples_dirs.slice(0, 0, cur_nr_samples_cpu);
    ray_samples_packed_valid.samples_z=samples_z.slice(0, 0, cur_nr_samples_cpu);
    ray_samples_packed_valid.samples_dt=samples_dt.slice(0, 0, cur_nr_samples_cpu);
    ray_samples_packed_valid.samples_sdf=samples_sdf.slice(0, 0, cur_nr_samples_cpu);
    ray_samples_packed_valid.ray_fixed_dt=ray_fixed_dt;
    ray_samples_packed_valid.ray_start_end_idx=ray_start_end_idx;

    return ray_samples_packed_valid;
}

void RaySamplesPacked::set_sdf(const torch::Tensor& sdf){
    samples_sdf=sdf;
    has_sdf=true;
}

    

