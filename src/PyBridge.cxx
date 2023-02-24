#include "hash_sdf/PyBridge.h"

#include <torch/extension.h>
#include "torch/torch.h"
#include "torch/csrc/utils/pybind.h"

//my stuff 
#include "hash_sdf/HashSDF.cuh"
#include "hash_sdf/Sphere.cuh"
#include "hash_sdf/OccupancyGrid.cuh"
#include "hash_sdf/VolumeRendering.cuh"
#include "hash_sdf/RaySampler.cuh"
#include "hash_sdf/RaySamplesPacked.cuh"
#include "hash_sdf/TrainParams.h"
// #include "hash_sdf/ModelParams.h"
// #include "hash_sdf/EvalParams.h"
#include "hash_sdf/NGPGui.h"

#include "easy_pbr/Viewer.h"


// https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html
// PYBIND11_MAKE_OPAQUE(std::vector<int>); //to be able to pass vectors by reference to functions and have things like push back actually work 
// PYBIND11_MAKE_OPAQUE(std::vector<float>, std::allocator<float> >);

namespace py = pybind11;




PYBIND11_MODULE(hash_sdf, m) {



   

    py::class_<HashSDF, std::shared_ptr<HashSDF>   > (m, "HashSDF")
    .def_static("create", &HashSDF::create<const std::shared_ptr<easy_pbr::Viewer>& > ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    .def_static("random_rays_from_reel", &HashSDF::random_rays_from_reel ) 
    .def_static("rays_from_reprojection_reel", &HashSDF::rays_from_reprojection_reel ) 
    .def_static("spherical_harmonics", &HashSDF::spherical_harmonics ) 
    .def_static("update_errors_of_matching_indices", &HashSDF::update_errors_of_matching_indices ) 
    .def_static("meshgrid3d", &HashSDF::meshgrid3d ) 
    .def_static("low_discrepancy2d_sampling", &HashSDF::low_discrepancy2d_sampling ) 
    #ifdef HSDF_WITH_GL
      .def("render_atributes",  &HashSDF::render_atributes )
    #endif
    ;

    py::class_<Sphere> (m, "Sphere")
    .def(py::init<const float, const Eigen::Vector3f>())
    .def("ray_intersection", &Sphere::ray_intersection ) 
    .def("rand_points_inside", &Sphere::rand_points_inside, py::arg("nr_points") ) 
    .def_readwrite("m_center_tensor", &Sphere::m_center_tensor ) 
    .def_readwrite("m_center", &Sphere::m_center ) 
    .def_readwrite("m_radius", &Sphere::m_radius ) 
    ;

    // py::class_<VoxelGrid> (m, "VoxelGrid")
    // .def(py::init<>())
    // .def_static("slice", &VoxelGrid::slice ) 
    // .def_static("splat", &VoxelGrid::splat) 
    // .def_static("upsample_grid", &VoxelGrid::upsample_grid) 
    // .def_static("get_nr_of_mips", &VoxelGrid::get_nr_of_mips ) 
    // .def_static("get_size_for_mip", &VoxelGrid::get_size_for_mip ) 
    // .def_static("get_size_downsampled_grid", &VoxelGrid::get_size_downsampled_grid ) 
    // .def_static("get_size_upsampled_grid", &VoxelGrid::get_size_upsampled_grid ) 
    // .def_static("compute_grid_points", &VoxelGrid::compute_grid_points ) 
    // .def_static("slice_cpu", &VoxelGrid::slice_cpu ) 
    // ;


    py::class_<OccupancyGrid> (m, "OccupancyGrid")
    .def(py::init<const int, const float, const Eigen::Vector3f>())
    .def_static("make_grid_values", &OccupancyGrid::make_grid_values ) 
    .def_static("make_grid_occupancy", &OccupancyGrid::make_grid_occupancy ) 
    .def("set_grid_values", &OccupancyGrid::set_grid_values ) 
    .def("set_grid_occupancy", &OccupancyGrid::set_grid_occupancy ) 
    .def("get_grid_values", &OccupancyGrid::get_grid_values ) 
    .def("get_grid_occupancy", &OccupancyGrid::get_grid_occupancy ) 
    .def("get_nr_voxels", &OccupancyGrid::get_nr_voxels ) 
    .def("get_nr_voxels_per_dim", &OccupancyGrid::get_nr_voxels_per_dim ) 
    .def("compute_grid_points", &OccupancyGrid::compute_grid_points ) 
    .def("compute_random_sample_of_grid_points", &OccupancyGrid::compute_random_sample_of_grid_points ) 
    // .def("create_cubes_for_occupied_voxels", &OccupancyGrid::create_cubes_for_occupied_voxels ) 
    .def("check_occupancy", &OccupancyGrid::check_occupancy ) 
    .def("update_with_density", &OccupancyGrid::update_with_density ) 
    .def("update_with_density_random_sample", &OccupancyGrid::update_with_density_random_sample ) 
    .def("update_with_sdf", &OccupancyGrid::update_with_sdf ) 
    .def("update_with_sdf_random_sample", &OccupancyGrid::update_with_sdf_random_sample ) 
    .def("compute_samples_in_occupied_regions", &OccupancyGrid::compute_samples_in_occupied_regions ) 
    .def("compute_first_sample_start_of_occupied_regions", &OccupancyGrid::compute_first_sample_start_of_occupied_regions ) 
    .def("advance_sample_to_next_occupied_voxel", &OccupancyGrid::advance_sample_to_next_occupied_voxel ) 
    ;

    py::class_<RaySamplesPacked> (m, "RaySamplesPacked")
    .def(py::init<const int, const int>())
    .def("get_valid_samples", &RaySamplesPacked::get_valid_samples ) 
    .def_readwrite("samples_pos",  &RaySamplesPacked::samples_pos )
    .def_readwrite("samples_pos_4d",  &RaySamplesPacked::samples_pos_4d )
    .def_readwrite("samples_dirs",  &RaySamplesPacked::samples_dirs )
    .def_readwrite("samples_z",  &RaySamplesPacked::samples_z )
    .def_readwrite("samples_dt",  &RaySamplesPacked::samples_dt )
    .def_readwrite("samples_sdf",  &RaySamplesPacked::samples_sdf )
    .def("set_sdf", &RaySamplesPacked::set_sdf ) 
    .def_readwrite("ray_start_end_idx",  &RaySamplesPacked::ray_start_end_idx )
    .def_readwrite("ray_fixed_dt",  &RaySamplesPacked::ray_fixed_dt )
    .def_readwrite("max_nr_samples",  &RaySamplesPacked::max_nr_samples )
    .def_readwrite("cur_nr_samples",  &RaySamplesPacked::cur_nr_samples )
    .def_readwrite("rays_have_equal_nr_of_samples",  &RaySamplesPacked::rays_have_equal_nr_of_samples )
    .def_readwrite("fixed_nr_of_samples_per_ray",  &RaySamplesPacked::fixed_nr_of_samples_per_ray )
    ;

    py::class_<VolumeRendering> (m, "VolumeRendering")
    .def(py::init<>())
    .def_static("volume_render_nerf", &VolumeRendering::volume_render_nerf ) 
    .def_static("compute_dt", &VolumeRendering::compute_dt ) 
    .def_static("cumprod_alpha2transmittance", &VolumeRendering::cumprod_alpha2transmittance ) 
    .def_static("integrate_rgb_and_weights", &VolumeRendering::integrate_rgb_and_weights ) 
    .def_static("sdf2alpha", &VolumeRendering::sdf2alpha ) 
    .def_static("sum_over_each_ray", &VolumeRendering::sum_over_each_ray ) 
    .def_static("cumsum_over_each_ray", &VolumeRendering::cumsum_over_each_ray ) 
    .def_static("compute_cdf", &VolumeRendering::compute_cdf )  
    .def_static("importance_sample", &VolumeRendering::importance_sample )  
    .def_static("combine_uniform_samples_with_imp", &VolumeRendering::combine_uniform_samples_with_imp )  
    .def_static("compact_ray_samples", &VolumeRendering::compact_ray_samples )  
    //backward passes
    .def_static("volume_render_nerf_backward", &VolumeRendering::volume_render_nerf_backward ) 
    .def_static("cumprod_alpha2transmittance_backward", &VolumeRendering::cumprod_alpha2transmittance_backward )  
    .def_static("integrate_rgb_and_weights_backward", &VolumeRendering::integrate_rgb_and_weights_backward )  
    .def_static("sum_over_each_ray_backward", &VolumeRendering::sum_over_each_ray_backward )  
    ;

    py::class_<RaySampler> (m, "RaySampler")
    .def(py::init<>())
    .def_static("compute_samples_bg", &RaySampler::compute_samples_bg ) 
    ;



    //TrainParams
    py::class_<TrainParams, std::shared_ptr<TrainParams>   > (m, "TrainParams", py::module_local())
    .def_static("create", &TrainParams::create<const std::string> ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    .def("dataset_name",  &TrainParams::dataset_name )
    .def("with_viewer",  &TrainParams::with_viewer )
    .def("with_visdom",  &TrainParams::with_visdom )
    .def("with_tensorboard",  &TrainParams::with_tensorboard )
    .def("with_wandb",  &TrainParams::with_wandb )
    .def("lr",  &TrainParams::lr )
    .def("weight_decay",  &TrainParams::weight_decay )
    .def("save_checkpoint",  &TrainParams::save_checkpoint )
    .def("checkpoint_path",  &TrainParams::checkpoint_path )
    ;

    // //EvalParams
    // py::class_<EvalParams, std::shared_ptr<EvalParams>   > (m, "EvalParams", py::module_local())
    // .def_static("create", &EvalParams::create<const std::string> ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    // .def("dataset_name",  &EvalParams::dataset_name )
    // .def("with_viewer",  &EvalParams::with_viewer )
    // .def("checkpoint_path",  &EvalParams::checkpoint_path )
    // .def("do_write_predictions",  &EvalParams::do_write_predictions )
    // .def("output_predictions_path",  &EvalParams::output_predictions_path )
    // ;

    // //ModelParams
    // py::class_<ModelParams, std::shared_ptr<ModelParams>   > (m, "ModelParams", py::module_local())
    // .def_static("create", &ModelParams::create<const std::string> ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    // .def("positions_mode",  &ModelParams::positions_mode )
    // .def("values_mode",  &ModelParams::values_mode )
    // .def("pointnet_channels_per_layer",  &ModelParams::pointnet_channels_per_layer )
    // .def("pointnet_start_nr_channels",  &ModelParams::pointnet_start_nr_channels )
    // .def("nr_downsamples",  &ModelParams::nr_downsamples )
    // .def("nr_blocks_down_stage",  &ModelParams::nr_blocks_down_stage )
    // .def("nr_blocks_bottleneck",  &ModelParams::nr_blocks_bottleneck )
    // .def("nr_blocks_up_stage",  &ModelParams::nr_blocks_up_stage )
    // .def("nr_levels_down_with_normal_resnet",  &ModelParams::nr_levels_down_with_normal_resnet )
    // .def("nr_levels_up_with_normal_resnet",  &ModelParams::nr_levels_up_with_normal_resnet )
    // .def("compression_factor",  &ModelParams::compression_factor )
    // .def("dropout_last_layer",  &ModelParams::dropout_last_layer )
    // // .def("experiment",  &ModelParams::experiment )
    // ;

    //NGPGui
    py::class_<NGPGui, std::shared_ptr<NGPGui>   > (m, "NGPGui", py::module_local())
    .def_static("create",  &NGPGui::create<const std::shared_ptr<easy_pbr::Viewer>& > ) //for templated methods like this one we need to explicitly 
    .def_readwrite("m_do_training",  &NGPGui::m_do_training )
    .def_readwrite("m_control_view",  &NGPGui::m_control_view )
    .def_readwrite("m_c2f_progress",  &NGPGui::m_c2f_progress )
    .def_readwrite("m_nr_samples_per_ray",  &NGPGui::m_nr_samples_per_ray )
    .def_readwrite("m_inv_s",  &NGPGui::m_inv_s )
    .def_readwrite("m_inv_s_min",  &NGPGui::m_inv_s_min )
    .def_readwrite("m_inv_s_max",  &NGPGui::m_inv_s_max )
    .def_readwrite("m_volsdf_beta",  &NGPGui::m_volsdf_beta )
    .def_readwrite("m_neus_cos_anneal",  &NGPGui::m_neus_cos_anneal )
    .def_readwrite("m_neus_variance",  &NGPGui::m_neus_variance )
    .def_readwrite("m_nerf_surface_beta",  &NGPGui::m_nerf_surface_beta )
    .def_readwrite("m_nerf_surface_std",  &NGPGui::m_nerf_surface_std )
    .def_readwrite("m_surface_prob_sigma",  &NGPGui::m_surface_prob_sigma )
    .def_readwrite("m_surface_prob_height",  &NGPGui::m_surface_prob_height )
    .def_readwrite("m_soft_opacity_sigma",  &NGPGui::m_soft_opacity_sigma )
    .def_readwrite("m_sphere_y_shift",  &NGPGui::m_sphere_y_shift )
    .def_readwrite("m_show_unisurf_weights",  &NGPGui::m_show_unisurf_weights )
    .def_readwrite("m_show_volsdf_weights",  &NGPGui::m_show_volsdf_weights )
    .def_readwrite("m_show_neus_weights",  &NGPGui::m_show_neus_weights )
    .def_readwrite("m_show_nerf_surface_weights",  &NGPGui::m_show_nerf_surface_weights )
    .def_readwrite("m_ray_origin_x_shift",  &NGPGui::m_ray_origin_x_shift )
    .def_readwrite("m_ray_origin_y_shift",  &NGPGui::m_ray_origin_y_shift )
    .def_readwrite("m_isolines_layer_z_coord",  &NGPGui::m_isolines_layer_z_coord )
    .def_readwrite("m_compute_full_layer",  &NGPGui::m_compute_full_layer )
    .def_readwrite("m_isoline_width",  &NGPGui::m_isoline_width )
    .def_readwrite("m_distance_between_isolines",  &NGPGui::m_distance_between_isolines )
    .def_readwrite("m_use_only_dense_grid",  &NGPGui::m_use_only_dense_grid )
    .def_readwrite("m_spp",  &NGPGui::m_spp )
    .def_readwrite("m_render_mitsuba",  &NGPGui::m_render_mitsuba )
    .def_readwrite("m_mitsuba_res_x",  &NGPGui::m_mitsuba_res_x )
    .def_readwrite("m_mitsuba_res_y",  &NGPGui::m_mitsuba_res_y )
    .def_readwrite("m_use_controlable_frame",  &NGPGui::m_use_controlable_frame )
    .def_readwrite("m_frame_idx_from_dataset",  &NGPGui::m_frame_idx_from_dataset )
    .def_readwrite("m_render_full_img",  &NGPGui::m_render_full_img )
    .def_readwrite("m_use_sphere_tracing",  &NGPGui::m_use_sphere_tracing )
    .def_readwrite("m_nr_iters_sphere_tracing",  &NGPGui::m_nr_iters_sphere_tracing )
    .def_readwrite("m_sphere_trace_agressiveness",  &NGPGui::m_sphere_trace_agressiveness )
    .def_readwrite("m_sphere_trace_threshold_converged",  &NGPGui::m_sphere_trace_threshold_converged )
    .def_readwrite("m_sphere_trace_push_in_gradient_dir",  &NGPGui::m_sphere_trace_push_in_gradient_dir )
    .def_readwrite("m_chunk_size",  &NGPGui::m_chunk_size )
    .def_readwrite("m_error_map_max",  &NGPGui::m_error_map_max )
    ;


}



