#include "permuto_sdf/PyBridge.h"

#include "torch/torch.h"
#include <torch/extension.h>


//my stuff 
#include "permuto_sdf/PermutoSDF.cuh"
#include "permuto_sdf/Sphere.cuh"
#include "permuto_sdf/OccupancyGrid.cuh"
#include "permuto_sdf/VolumeRendering.cuh"
#include "permuto_sdf/RaySampler.cuh"
#include "permuto_sdf/RaySamplesPacked.cuh"
#include "permuto_sdf/TrainParams.h"
#include "permuto_sdf/NGPGui.h"

#include "easy_pbr/Viewer.h"


// https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html
// PYBIND11_MAKE_OPAQUE(std::vector<int>); //to be able to pass vectors by reference to functions and have things like push back actually work 
// PYBIND11_MAKE_OPAQUE(std::vector<float>, std::allocator<float> >);

namespace py = pybind11;




PYBIND11_MODULE(permuto_sdf, m) {



   

    py::class_<PermutoSDF, std::shared_ptr<PermutoSDF>   > (m, "PermutoSDF")
    .def_static("create", &PermutoSDF::create<const std::shared_ptr<easy_pbr::Viewer>& > ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    .def_static("random_rays_from_reel", &PermutoSDF::random_rays_from_reel ) 
    .def_static("rays_from_reprojection_reel", &PermutoSDF::rays_from_reprojection_reel ) 
    .def_static("spherical_harmonics", &PermutoSDF::spherical_harmonics ) 
    .def_static("update_errors_of_matching_indices", &PermutoSDF::update_errors_of_matching_indices ) 
    .def_static("meshgrid3d", &PermutoSDF::meshgrid3d ) 
    .def_static("low_discrepancy2d_sampling", &PermutoSDF::low_discrepancy2d_sampling ) 
    #ifdef HSDF_WITH_GL
      .def("render_atributes",  &PermutoSDF::render_atributes )
    #endif
    ;

    py::class_<Sphere> (m, "Sphere")
    .def(py::init<const float, const Eigen::Vector3f>())
    .def("ray_intersection", &Sphere::ray_intersection ) 
    .def("rand_points_inside", &Sphere::rand_points_inside, py::arg("nr_points") ) 
    .def("check_point_inside_primitive", &Sphere::check_point_inside_primitive ) 
    .def_readwrite("m_center_tensor", &Sphere::m_center_tensor ) 
    .def_readwrite("m_center", &Sphere::m_center ) 
    .def_readwrite("m_radius", &Sphere::m_radius ) 
    ;


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
    .def("compact_to_valid_samples", &RaySamplesPacked::compact_to_valid_samples ) 
    .def("compute_exact_nr_samples", &RaySamplesPacked::compute_exact_nr_samples ) 
    .def("initialize_with_one_sample_per_ray", &RaySamplesPacked::initialize_with_one_sample_per_ray ) 
    .def("set_sdf", &RaySamplesPacked::set_sdf ) 
    .def("remove_sdf", &RaySamplesPacked::remove_sdf ) 
    .def_readwrite("samples_pos",  &RaySamplesPacked::samples_pos )
    .def_readwrite("samples_pos_4d",  &RaySamplesPacked::samples_pos_4d )
    .def_readwrite("samples_dirs",  &RaySamplesPacked::samples_dirs )
    .def_readwrite("samples_z",  &RaySamplesPacked::samples_z )
    .def_readwrite("samples_dt",  &RaySamplesPacked::samples_dt )
    .def_readwrite("samples_sdf",  &RaySamplesPacked::samples_sdf )
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
    .def_static("integrate_with_weights", &VolumeRendering::integrate_with_weights ) 
    .def_static("sdf2alpha", &VolumeRendering::sdf2alpha ) 
    .def_static("sum_over_each_ray", &VolumeRendering::sum_over_each_ray ) 
    .def_static("cumsum_over_each_ray", &VolumeRendering::cumsum_over_each_ray ) 
    .def_static("compute_cdf", &VolumeRendering::compute_cdf )  
    .def_static("importance_sample", &VolumeRendering::importance_sample )  
    .def_static("combine_uniform_samples_with_imp", &VolumeRendering::combine_uniform_samples_with_imp )  
    //backward passes
    .def_static("volume_render_nerf_backward", &VolumeRendering::volume_render_nerf_backward ) 
    .def_static("cumprod_alpha2transmittance_backward", &VolumeRendering::cumprod_alpha2transmittance_backward )  
    .def_static("integrate_with_weights_backward", &VolumeRendering::integrate_with_weights_backward )  
    .def_static("sum_over_each_ray_backward", &VolumeRendering::sum_over_each_ray_backward )  
    ;

    py::class_<RaySampler> (m, "RaySampler")
    .def(py::init<>())
    .def_static("compute_samples_fg", &RaySampler::compute_samples_fg ) 
    .def_static("compute_samples_bg", &RaySampler::compute_samples_bg ) 
    ;



    //TrainParams
    py::class_<TrainParams, std::shared_ptr<TrainParams>   > (m, "TrainParams", py::module_local())
    .def_static("create", &TrainParams::create<const std::string> ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    .def("with_visdom",  &TrainParams::with_visdom )
    .def("with_tensorboard",  &TrainParams::with_tensorboard )
    .def("with_wandb",  &TrainParams::with_wandb )
    .def("save_checkpoint",  &TrainParams::save_checkpoint )
    //setters
    .def("set_with_visdom",  &TrainParams::set_with_visdom )
    .def("set_with_tensorboard",  &TrainParams::set_with_tensorboard )
    .def("set_with_wandb",  &TrainParams::set_with_wandb )
    .def("set_save_checkpoint",  &TrainParams::set_save_checkpoint )
    ;


    //NGPGui
    py::class_<NGPGui, std::shared_ptr<NGPGui>   > (m, "NGPGui", py::module_local())
    .def_static("create",  &NGPGui::create<const std::shared_ptr<easy_pbr::Viewer>& > ) //for templated methods like this one we need to explicitly 
    .def_readwrite("m_do_training",  &NGPGui::m_do_training )
    .def_readwrite("m_control_view",  &NGPGui::m_control_view )
    .def_readwrite("m_time_val",  &NGPGui::m_time_val )
    .def_readwrite("m_c2f_progress",  &NGPGui::m_c2f_progress )
    .def_readwrite("m_isolines_layer_z_coord",  &NGPGui::m_isolines_layer_z_coord )
    .def_readwrite("m_compute_full_layer",  &NGPGui::m_compute_full_layer )
    .def_readwrite("m_isoline_width",  &NGPGui::m_isoline_width )
    .def_readwrite("m_distance_between_isolines",  &NGPGui::m_distance_between_isolines )
    .def_readwrite("m_use_controlable_frame",  &NGPGui::m_use_controlable_frame )
    .def_readwrite("m_frame_idx_from_dataset",  &NGPGui::m_frame_idx_from_dataset )
    .def_readwrite("m_render_full_img",  &NGPGui::m_render_full_img )
    .def_readwrite("m_use_sphere_tracing",  &NGPGui::m_use_sphere_tracing )
    .def_readwrite("m_nr_iters_sphere_tracing",  &NGPGui::m_nr_iters_sphere_tracing )
    .def_readwrite("m_sphere_trace_agressiveness",  &NGPGui::m_sphere_trace_agressiveness )
    .def_readwrite("m_sphere_trace_threshold_converged",  &NGPGui::m_sphere_trace_threshold_converged )
    .def_readwrite("m_chunk_size",  &NGPGui::m_chunk_size )
    ;


}



