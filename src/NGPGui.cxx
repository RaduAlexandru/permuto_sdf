#include "hash_sdf/NGPGui.h"

//c++

//easypbr
#include "easy_pbr/Gui.h"
#include "easy_pbr/Viewer.h"


//loguru
// #define LOGURU_REPLACE_GLOG 1
// #include <loguru.hpp> //needs to be added after torch.h otherwise loguru stops printing for some reason




NGPGui::NGPGui(const std::shared_ptr<easy_pbr::Viewer>& view):
    m_do_training(true),
    m_control_view(false),
    m_time_val(0.0),
    m_c2f_progress(0),
    // m_nr_samples_per_ray(1000),
    // m_inv_s(512),
    // m_inv_s_min(512),
    // m_inv_s_max(1024),
    // m_volsdf_beta(0.01),
    // m_neus_cos_anneal(1.0),
    // m_neus_variance(0.3),
    // m_nerf_surface_beta(0.01),
    // m_nerf_surface_std(0.3),
    // m_surface_prob_sigma(0.1),
    // m_surface_prob_height(0.95),
    // m_soft_opacity_sigma(22.0),
    // m_sphere_y_shift(0.0),
    // m_show_unisurf_weights(false),
    // m_show_volsdf_weights(false),
    // m_show_neus_weights(false),
    // m_show_nerf_surface_weights(false),
    // m_ray_origin_x_shift(0.0),
    // m_ray_origin_y_shift(0.0),
    // m_isolines_layer_z_coord(0.0),
    // m_compute_full_layer(false),
    // m_isoline_width(0.005),
    // m_distance_between_isolines(0.06),
    // m_use_only_dense_grid(false),
    // m_spp(50),
    // m_render_mitsuba(false),
    // m_mitsuba_res_x(500),
    // m_mitsuba_res_y(500),
    // m_use_controlable_frame(true),
    // m_frame_idx_from_dataset(0),
    // m_render_full_img(false),
    // m_use_sphere_tracing(true),
    // m_nr_iters_sphere_tracing(1),
    // m_sphere_trace_agressiveness(1.0),
    // m_sphere_trace_threshold_converged(0.001),
    // m_sphere_trace_push_in_gradient_dir(0.0),
    // m_chunk_size(2000),
    // m_error_map_max(0.00001),
    m_view(view)
    {


    install_callbacks(view);

}

NGPGui::~NGPGui(){
    // LOG(WARNING) << "Deleting HairReconGUI";
}

void NGPGui::install_callbacks(const std::shared_ptr<easy_pbr::Viewer>& view){
    //pre draw functions (can install multiple functions and they will be called in order)

    //post draw functions
    view->add_callback_post_draw( [this]( easy_pbr::Viewer& v ) -> void{ this->post_draw(v); }  );
}


void NGPGui::post_draw(easy_pbr::Viewer& view){


    //draw some gui elements
    ImGuiWindowFlags window_flags = 0;
    ImGui::Begin("NGPGUI", nullptr, window_flags);


    // ImGui::Checkbox("do_training", &m_do_training);
    ImGui::Checkbox("control_view", &m_control_view);
    ImGui::SliderFloat("time_val", &m_time_val, 0.0, 1.0) ;

    //progress bar for c2f
    ImGui::ProgressBar(m_c2f_progress, ImVec2(0.0f, 0.0f));
    // ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
    // ImGui::Text("coarse2fine");

    // if (ImGui::CollapsingHeader("Render from Frame") ) {
    //     ImGui::Checkbox("use_controlable_frame", &m_use_controlable_frame);
    //     ImGui::Checkbox("m_use_sphere_tracing", &m_use_sphere_tracing);
    //     ImGui::SliderInt("nr_iters_sphere_tracing", &m_nr_iters_sphere_tracing, 1, 10) ;
    //     ImGui::SliderFloat("sphere_trace_agressiveness", &m_sphere_trace_agressiveness, 0.0, 3.0) ;
    //     ImGui::SliderFloat("sphere_trace_threshold_converged", &m_sphere_trace_threshold_converged, 0.0, 0.1) ;
    //     ImGui::SliderFloat("sphere_trace_push_in_gradient_dir", &m_sphere_trace_push_in_gradient_dir, -0.1, 0.1) ;
    //     ImGui::SliderInt("chunk_size", &m_chunk_size, 0, 20000) ;
    //     ImGui::SliderInt("frame_idx_from_dataset", &m_frame_idx_from_dataset, 0, 100) ;
    //     ImGui::SliderFloat("isolines_layer_z_coord", &m_isolines_layer_z_coord, -1.0f, 1.0f) ;
    //     ImGui::SliderFloat("isoline_width", &m_isoline_width, 0.0f, 0.05f) ;
    //     ImGui::SliderFloat("distance_between_isolines", &m_distance_between_isolines, 0.0f, 0.3f) ;
    //     if (ImGui::Button("m_compute_full_layer") ) {
    //         m_compute_full_layer=true;
    //     }
    //     if (ImGui::Button("Render Full Image") ) {
    //         m_render_full_img=true;
    //     }
    // }

    // //plotting stuff
    // ImGui::Checkbox("show_unisurf_weights", &m_show_unisurf_weights);
    // ImGui::Checkbox("show_volsdf_weights", &m_show_volsdf_weights);
    // ImGui::Checkbox("show_neus_weights", &m_show_neus_weights);
    // ImGui::Checkbox("show_nerf_surface_weights", &m_show_nerf_surface_weights);
    // if (ImGui::CollapsingHeader("Ray") ) {
    //     ImGui::SliderInt("nr_samples_per_ray", &m_nr_samples_per_ray, 10, 10000) ;
    //     ImGui::SliderFloat("inv_s", &m_inv_s, 0.0f, 2048.0f) ;
    //     ImGui::SliderFloat("inv_s_min", &m_inv_s_min, 0.0f, 2048.0f) ;
    //     ImGui::SliderFloat("inv_s_max", &m_inv_s_max, 0.0f, 2048.0f) ;
    //     // ImGui::SliderFloat("inv_s_min", &m_inv_s_min, -10.0f, 10.0f) ;
    //     // ImGui::SliderFloat("inv_s_max", &m_inv_s_max, -10.0f, 10.0f) ;
    //     ImGui::SliderFloat("ray_origin_x_shift", &m_ray_origin_x_shift, -10.0f, 10.0f) ;
    //     ImGui::SliderFloat("ray_origin_y_shift", &m_ray_origin_y_shift, -10.0f, 10.0f) ;
    //     ImGui::SliderFloat("sphere_y_shift", &m_sphere_y_shift, -2.5f, 2.5f) ;
    //     ImGui::SliderFloat("surface_prob_sigma", &m_surface_prob_sigma, 0.001f, 2.0f) ;
    //     ImGui::SliderFloat("surface_prob_height", &m_surface_prob_height, 0.1f, 1.0f) ;
    //     ImGui::SliderFloat("soft_opacity_sigma", &m_soft_opacity_sigma, 0.1f, 55.0f) ;
    // }
    // ImGui::SliderFloat("volsdf_beta", &m_volsdf_beta, 0.001f, 0.3f) ;
    // ImGui::SliderFloat("neus_cos_anneal", &m_neus_cos_anneal, 0.0f, 1.0f) ;
    // ImGui::SliderFloat("neus_variance", &m_neus_variance, 0.00001f, 1.0f) ;
    // ImGui::SliderFloat("nerf_surface_beta", &m_nerf_surface_beta, 0.001f, 2.0f) ;
    // ImGui::SliderFloat("nerf_surface_std", &m_nerf_surface_std, 0.001f, 2.0f) ;

    // if (ImGui::CollapsingHeader("Visualize isolines") ) {
    //     ImGui::SliderFloat("isolines_layer_z_coord", &m_isolines_layer_z_coord, -1.0f, 1.0f) ;
    //     ImGui::SliderFloat("isoline_width", &m_isoline_width, 0.0f, 0.05f) ;
    //     ImGui::SliderFloat("distance_between_isolines", &m_distance_between_isolines, 0.0f, 0.3f) ;
    //     if (ImGui::Button("m_compute_full_layer") ) {
    //         m_compute_full_layer=true;
    //     }
    // }

    // if (ImGui::CollapsingHeader("Mitsuba") ) {
    //     ImGui::SliderInt("spp", &m_spp, 1, 256) ;
    //     ImGui::SliderInt("m_mitsuba_res_x", &m_mitsuba_res_x, 50, 1920) ;
    //     ImGui::SliderInt("m_mitsuba_res_y", &m_mitsuba_res_y, 50, 1080) ;
    //     if (ImGui::Button("render") ) {
    //         m_render_mitsuba=true;
    //     }
        
    // }

    // if (ImGui::CollapsingHeader("Misc") ) {
    //     ImGui::SliderFloat("m_error_map_max", &m_error_map_max, 0.0f, 0.00005f) ;
    // }

    // ImGui::Checkbox("use_only_dense_grid", &m_use_only_dense_grid);

    
    ImGui::End();

    // m_iter++;
}

