#include "permuto_sdf/OccupancyGrid.cuh"

//c++
// #include <string>

#include "UtilsPytorch.h" //contains torch so it has to be added BEFORE any other include because the other ones might include loguru which gets screwed up if torch was included before it

//my stuff
#include "permuto_sdf/OccupancyGridGPU.cuh"


using torch::Tensor;
using namespace radu::utils;



template <typename T>
T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

pcg32 OccupancyGrid::m_rng;


//CPU code that calls the kernels
OccupancyGrid::OccupancyGrid(const int nr_voxels_per_dim, const float grid_extent, const Eigen::Vector3f grid_translation):
    m_nr_voxels_per_dim(nr_voxels_per_dim),
    m_grid_extent(grid_extent),
    m_grid_translation(grid_translation)
    {

        //make some things to tensors
        m_grid_translation_tensor=eigen2tensor(m_grid_translation).squeeze(1).cuda();

        //make a grid which stores the floating point value occupancy
        //is starts at 1 showing that it is all occupied at the beggining
        m_grid_values=make_grid_values(nr_voxels_per_dim);
        m_grid_occupancy=make_grid_occupancy(nr_voxels_per_dim);

}

OccupancyGrid::~OccupancyGrid(){
}

int OccupancyGrid::get_nr_voxels(){
    return m_nr_voxels_per_dim*m_nr_voxels_per_dim*m_nr_voxels_per_dim; 
}
int OccupancyGrid::get_nr_voxels_per_dim(){
    return m_nr_voxels_per_dim;
}
torch::Tensor OccupancyGrid::get_grid_values(){
    return m_grid_values;
}
torch::Tensor OccupancyGrid::get_grid_occupancy(){
    return m_grid_occupancy;
}


void OccupancyGrid::set_grid_values(const torch::Tensor& grid_values){
    m_grid_values=grid_values;
}
void OccupancyGrid::set_grid_occupancy(const torch::Tensor& grid_occupancy){
    m_grid_occupancy=grid_occupancy;
}


torch::Tensor OccupancyGrid::make_grid_values(const int nr_voxels_per_dim){
    CHECK(nr_voxels_per_dim%2==0) <<"We are expecting an even number of voxels because we consider the value of the voxel to live a the center of the cube. We need to have even numbers because we are using morton codes";
    //https://stackoverflow.com/a/108360
    CHECK((nr_voxels_per_dim & (nr_voxels_per_dim - 1)) == 0)  << "Nr of voxels should be power of 2 because we are using morton codes";
    // torch::Tensor grid=torch::empty({nr_voxels_per_dim, nr_voxels_per_dim, nr_voxels_per_dim, val_dim },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    torch::Tensor grid=torch::ones({ nr_voxels_per_dim* nr_voxels_per_dim* nr_voxels_per_dim },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );

    return grid;
}

torch::Tensor OccupancyGrid::make_grid_occupancy(const int nr_voxels_per_dim){
    CHECK(nr_voxels_per_dim%2==0) <<"We are expecting an even number of voxels because we consider the value of the voxel to live a the center of the cube. We need to have even numbers because we are using morton codes";
    //https://stackoverflow.com/a/108360
    CHECK((nr_voxels_per_dim & (nr_voxels_per_dim - 1)) == 0)  << "Nr of voxels should be power of 2 because we are using morton codes";
    // torch::Tensor grid=torch::empty({nr_voxels_per_dim, nr_voxels_per_dim, nr_voxels_per_dim, val_dim },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    torch::Tensor grid=torch::ones({ nr_voxels_per_dim* nr_voxels_per_dim* nr_voxels_per_dim },  torch::dtype(torch::kBool).device(torch::kCUDA, 0)  );

    return grid;
}

torch::Tensor OccupancyGrid::compute_grid_points(const bool randomize_position){

    //make some positions in the range [-1,1], as many as new_nr_voxels_per_dim

    int nr_voxels=get_nr_voxels();

    // TIME_START("make_grid_points");
    torch::Tensor grid_points=torch::empty({ m_nr_voxels_per_dim* m_nr_voxels_per_dim* m_nr_voxels_per_dim,3 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    // TIME_END("make_grid_points");

    const dim3 blocks = { (unsigned int)div_round_up(nr_voxels, BLOCK_SIZE), 1, 1 }; 


    OccupancyGridGPU::compute_grid_points_gpu<<<blocks, BLOCK_SIZE>>>(
                nr_voxels,
                m_nr_voxels_per_dim,
                m_grid_extent,
                m_grid_translation_tensor.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                m_rng,
                randomize_position,
                //output
                grid_points.packed_accessor32<float,2,torch::RestrictPtrTraits>()
            );

    if(randomize_position){
        m_rng.advance();
    }

    return grid_points;
}

std::tuple<torch::Tensor,torch::Tensor> OccupancyGrid::compute_random_sample_of_grid_points(const int nr_voxels_to_select, const bool randomize_position){

    int nr_voxels_total=get_nr_voxels();

    torch::Tensor grid_points=torch::empty({ nr_voxels_to_select,3 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    //create also a tensor of nr_points,1 saying for each point, which voxel does it correspond to 
    torch::Tensor point_indices=torch::randint(0, nr_voxels_total, { nr_voxels_to_select },  torch::dtype(torch::kInt32).device(torch::kCUDA, 0)  );

    const dim3 blocks = { (unsigned int)div_round_up(nr_voxels_to_select, BLOCK_SIZE), 1, 1 }; 


    OccupancyGridGPU::compute_random_sample_of_grid_points_gpu<<<blocks, BLOCK_SIZE>>>(
                nr_voxels_to_select,
                m_nr_voxels_per_dim,
                m_grid_extent,
                m_grid_translation_tensor.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                point_indices.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                m_rng,
                randomize_position,
                //output
                grid_points.packed_accessor32<float,2,torch::RestrictPtrTraits>()
            );

    if(randomize_position){
        m_rng.advance();
    }

    return std::make_tuple(grid_points, point_indices);

}



RaySamplesPacked OccupancyGrid::compute_samples_in_occupied_regions(const torch::Tensor& ray_origins, const torch::Tensor& ray_dirs, const torch::Tensor& ray_t_entry, const torch::Tensor& ray_t_exit, const float min_dist_between_samples, const int max_nr_samples_per_ray, const bool jitter_samples){
    int nr_rays=ray_origins.size(0);
    // VLOG(1) << "nr rays " << nr_rays;

    int nr_samples_maximum=1024*1024*2; //we assume a maximum nr of rays and maximum of samples per ray
    RaySamplesPacked ray_samples_packed(nr_rays, nr_samples_maximum);


    //fill the samples
    const dim3 blocks = { (unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1 }; 


    // TIME_START("compute_samples_packed_cuda")
    OccupancyGridGPU::compute_samples_in_occupied_regions_gpu<<<blocks, BLOCK_SIZE>>>(
                nr_rays,
                m_nr_voxels_per_dim,
                m_grid_extent,
                m_grid_translation_tensor.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                ray_origins.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_dirs.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_t_entry.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_t_exit.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                m_grid_occupancy.packed_accessor32<bool,1,torch::RestrictPtrTraits>(),
                min_dist_between_samples,
                max_nr_samples_per_ray,
                ray_samples_packed.max_nr_samples,
                m_rng,
                jitter_samples,
                //output
                ray_samples_packed.samples_pos.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.samples_dirs.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.samples_z.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.samples_dt.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.ray_fixed_dt.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                // ray_samples_packed.max_nr_samples.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                ray_samples_packed.cur_nr_samples.packed_accessor32<int,1,torch::RestrictPtrTraits>()
            );
    // TIME_END("compute_samples_packed_cuda")

    if(jitter_samples){
        m_rng.advance();
    }

    return ray_samples_packed;
}

RaySamplesPacked OccupancyGrid::compute_first_sample_start_of_occupied_regions(const torch::Tensor& ray_origins, const torch::Tensor& ray_dirs, const torch::Tensor& ray_t_entry, const torch::Tensor& ray_t_exit){
    int nr_rays=ray_origins.size(0);
    // VLOG(1) << "nr rays " << nr_rays;

    int nr_samples_maximum=1024*1024*2; //we assume a maximum nr of rays and maximum of samples per ray
    RaySamplesPacked ray_samples_packed(nr_rays, nr_samples_maximum);


    //fill the samples
    const dim3 blocks = { (unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1 }; 


    OccupancyGridGPU::compute_first_sample_start_of_occupied_regions_gpu<<<blocks, BLOCK_SIZE>>>(
                nr_rays,
                m_nr_voxels_per_dim,
                m_grid_extent,
                m_grid_translation_tensor.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                ray_origins.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_dirs.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_t_entry.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_t_exit.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                m_grid_occupancy.packed_accessor32<bool,1,torch::RestrictPtrTraits>(),
                ray_samples_packed.max_nr_samples,
                // m_rng,
                // jitter_samples,
                //output
                ray_samples_packed.samples_pos.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.samples_dirs.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.samples_z.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.samples_dt.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.ray_fixed_dt.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                ray_samples_packed.ray_start_end_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                // ray_samples_packed.max_nr_samples.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                ray_samples_packed.cur_nr_samples.packed_accessor32<int,1,torch::RestrictPtrTraits>()
            );

    // if(jitter_samples){
    //     m_rng.advance();
    // }

    return ray_samples_packed;
}

std::tuple<torch::Tensor,torch::Tensor> OccupancyGrid::advance_sample_to_next_occupied_voxel(const torch::Tensor& samples_dirs, const torch::Tensor& samples_pos){

    int nr_points=samples_pos.size(0);



    //fill the samples
    const dim3 blocks = { (unsigned int)div_round_up(nr_points, BLOCK_SIZE), 1, 1 }; 

    torch::Tensor new_samples_pos=samples_pos;
    torch::Tensor is_within_bounds=torch::ones({ nr_points,1 },  torch::dtype(torch::kBool).device(torch::kCUDA, 0)  );


    OccupancyGridGPU::advance_sample_to_next_occupied_voxel_gpu<<<blocks, BLOCK_SIZE>>>(
                nr_points,
                m_nr_voxels_per_dim,
                m_grid_extent,
                m_grid_translation_tensor.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                // samples_origins.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                samples_dirs.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                samples_pos.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                m_grid_occupancy.packed_accessor32<bool,1,torch::RestrictPtrTraits>(),
                // m_rng,
                // jitter_samples,
                //output
                new_samples_pos.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                is_within_bounds.packed_accessor32<bool,2,torch::RestrictPtrTraits>()
            );

    // if(jitter_samples){
    //     m_rng.advance();
    // }

    return std::make_tuple(new_samples_pos, is_within_bounds);

}

torch::Tensor OccupancyGrid::check_occupancy(const torch::Tensor& points){

    CHECK(points.scalar_type()==at::kFloat) << "positions should be of type float";
    CHECK(points.dim()==2) << "positions should have dim 2 correspondin to HW. However it has sizes" << points.sizes();

    int nr_points=points.size(0);

    torch::Tensor occupancy_value=torch::ones({ nr_points,1 },  torch::dtype(torch::kBool).device(torch::kCUDA, 0)  );

    const dim3 blocks = { (unsigned int)div_round_up(nr_points, BLOCK_SIZE), 1, 1 }; 


    OccupancyGridGPU::check_occupancy_gpu<<<blocks, BLOCK_SIZE>>>(
                nr_points,
                m_nr_voxels_per_dim,
                m_grid_extent,
                m_grid_translation_tensor.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                m_grid_occupancy.packed_accessor32<bool,1,torch::RestrictPtrTraits>(),
                points.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                //output
                occupancy_value.packed_accessor32<bool,2,torch::RestrictPtrTraits>()
            );

  
    return occupancy_value;

}


void OccupancyGrid::update_with_density(const torch::Tensor& density, const float decay, const float occupancy_tresh){
    
    CHECK(density.dim()==2) << "density should have dim 2 correspondin to nr_pointsx1. However it has sizes" << density.sizes();
    CHECK(decay<1.0) <<"We except the decay to be <1.0 but it is " << decay;


    int nr_voxels=get_nr_voxels();


    const dim3 blocks = { (unsigned int)div_round_up(nr_voxels, BLOCK_SIZE), 1, 1 }; 


    OccupancyGridGPU::update_with_density_gpu<<<blocks, BLOCK_SIZE>>>(
                nr_voxels,
                density.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                m_nr_voxels_per_dim,
                decay,
                occupancy_tresh,
                // check_neighbours_density,
                // m_grid_full_values.packed_accessor32<float,3,torch::RestrictPtrTraits>()
                m_grid_values.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                m_grid_occupancy.packed_accessor32<bool,1,torch::RestrictPtrTraits>()
            );

}

void OccupancyGrid::update_with_density_random_sample(const torch::Tensor& point_indices, const torch::Tensor& density, const float decay, const float occupancy_tresh){

    CHECK(density.dim()==2) << "density should have dim 2 correspondin to nr_pointsx1. However it has sizes" << density.sizes();
    CHECK(decay<1.0) <<"We except the decay to be <1.0 but it is " << decay;
    CHECK(point_indices.dim()==1) << "point_indices should have dim 1 correspondin to nr_points. However it has sizes" << point_indices.sizes();


    int nr_points=point_indices.size(0);


    const dim3 blocks = { (unsigned int)div_round_up(nr_points, BLOCK_SIZE), 1, 1 }; 


    OccupancyGridGPU::update_with_density_random_sample_gpu<<<blocks, BLOCK_SIZE>>>(
                nr_points,
                density.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                m_nr_voxels_per_dim,
                point_indices.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                decay,
                occupancy_tresh,
                // check_neighbours_density,
                // m_grid_full_values.packed_accessor32<float,3,torch::RestrictPtrTraits>()
                m_grid_values.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                m_grid_occupancy.packed_accessor32<bool,1,torch::RestrictPtrTraits>()
            );

}

void OccupancyGrid::update_with_sdf(const torch::Tensor& sdf, const float inv_s, const float max_eikonal_abs, const float occupancy_thresh){

    CHECK(sdf.dim()==2) << "density should have dim 2 correspondin to nr_pointsx1. However it has sizes" << sdf.sizes();
    // CHECK(decay<1.0) <<"We except the decay to be <1.0 but it is " << decay;


    int nr_voxels=get_nr_voxels();

    const dim3 blocks = { (unsigned int)div_round_up(nr_voxels, BLOCK_SIZE), 1, 1 }; 


    OccupancyGridGPU::update_with_sdf_gpu<<<blocks, BLOCK_SIZE>>>(
                nr_voxels,
                sdf.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                m_grid_extent,
                m_nr_voxels_per_dim,
                inv_s,
                max_eikonal_abs,
                occupancy_thresh,
                m_grid_values.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                m_grid_occupancy.packed_accessor32<bool,1,torch::RestrictPtrTraits>()
            );

}

// void OccupancyGrid::update_with_sdf_random_sample(const torch::Tensor& point_indices, const torch::Tensor& sdf, const float inv_s, const float occupancy_thresh){
void OccupancyGrid::update_with_sdf_random_sample(const torch::Tensor& point_indices, const torch::Tensor& sdf, const torch::Tensor& inv_s, const float occupancy_thresh){

    CHECK(sdf.dim()==2) << "density should have dim 2 correspondin to nr_pointsx1. However it has sizes" << sdf.sizes();
    // CHECK(decay<1.0) <<"We except the decay to be <1.0 but it is " << decay;
    CHECK(point_indices.dim()==1) << "point_indices should have dim 1 correspondin to nr_points. However it has sizes" << point_indices.sizes();
    CHECK(inv_s.size(0)==1 && inv_s.dim()==1) << "Inv_s should be a tensor of 1 but it has sizes: " << inv_s.sizes();


    int nr_points=point_indices.size(0);


    const dim3 blocks = { (unsigned int)div_round_up(nr_points, BLOCK_SIZE), 1, 1 }; 


    OccupancyGridGPU::update_with_sdf_random_sample_gpu<<<blocks, BLOCK_SIZE>>>(
                nr_points,
                sdf.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                m_grid_extent,
                m_nr_voxels_per_dim,
                point_indices.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                // inv_s,
                inv_s.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                occupancy_thresh,
                m_grid_values.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                m_grid_occupancy.packed_accessor32<bool,1,torch::RestrictPtrTraits>()
            );

}

