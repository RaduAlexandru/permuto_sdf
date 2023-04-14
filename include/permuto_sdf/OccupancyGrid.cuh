#pragma once



#include "torch/torch.h"

#include <Eigen/Core>


#include "permuto_sdf/pcg32.h"

#include "permuto_sdf/RaySamplesPacked.cuh"


class OccupancyGrid{
public:
    OccupancyGrid(const int nr_voxels_per_dim, const float grid_extent, const Eigen::Vector3f grid_translation);
    ~OccupancyGrid();


    static torch::Tensor make_grid_values(const int nr_voxels_per_dim);
    static torch::Tensor make_grid_occupancy(const int nr_voxels_per_dim);
    torch::Tensor get_grid_values();
    torch::Tensor get_grid_occupancy();
    void set_grid_values(const torch::Tensor& grid_values);
    void set_grid_occupancy(const torch::Tensor& grid_occupancy);
    int get_nr_voxels();
    int get_nr_voxels_per_dim();
    

    torch::Tensor compute_grid_points(const bool randomize_position); //makes point at the position where the values lives. So actually it return the corners of the cubes because the values live at the corners
    std::tuple<torch::Tensor,torch::Tensor> compute_random_sample_of_grid_points(const int nr_of_voxels_to_select, const bool randomize_position);
    RaySamplesPacked compute_samples_in_occupied_regions(const torch::Tensor& ray_origins, const torch::Tensor& ray_dirs, const torch::Tensor& ray_t_entry, const torch::Tensor& ray_t_exit, const float min_dist_between_samples, const int max_nr_samples_per_ray, const bool jitter_samples); //goes through the occupies regions and creates samples 
    RaySamplesPacked compute_first_sample_start_of_occupied_regions(const torch::Tensor& ray_origins, const torch::Tensor& ray_dirs, const torch::Tensor& ray_t_entry, const torch::Tensor& ray_t_exit); //creates one sample only at the beggining of the occupied space, useful for sphere tracing
    torch::Tensor check_occupancy(const torch::Tensor& points);
    std::tuple<torch::Tensor,torch::Tensor> advance_sample_to_next_occupied_voxel(const torch::Tensor& samples_dirs, const torch::Tensor& samples_pos);
    

    void update_with_density(const torch::Tensor& density, const float decay, const float occupancy_tresh);
    void update_with_density_random_sample(const torch::Tensor& point_indices, const torch::Tensor& density, const float decay, const float occupancy_tresh);
    void update_with_sdf(const torch::Tensor& sdf, const float inv_s, const float max_eikonal_abs, const float occupancy_thresh);
    // void update_with_sdf_random_sample(const torch::Tensor& point_indices, const torch::Tensor& sdf, const float inv_s, const float occupancy_thresh);
    void update_with_sdf_random_sample(const torch::Tensor& point_indices, const torch::Tensor& sdf, const torch::Tensor& inv_s, const float occupancy_thresh);

    int m_nr_voxels_per_dim;
    float m_grid_extent; //size of the whole grid in any one dimension. For example if it's 1.0 then the cube has a side length of 1
    Eigen::Vector3f m_grid_translation;
    torch::Tensor m_grid_translation_tensor;
    
    torch::Tensor m_grid_values; //for each element in the grid,  we store the full value, either density or sdf
    torch::Tensor m_grid_occupancy; //just booleans saying if the voxels are occupied or not 

    static pcg32 m_rng;

private:
    

  
};
