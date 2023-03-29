#include "permuto_sdf/NGPGui.h"

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
    m_isolines_layer_z_coord(0.0),
    m_compute_full_layer(false),
    m_isoline_width(0.005),
    m_distance_between_isolines(0.03),
    m_render_full_img(false),
    m_use_controlable_frame(true),
    m_frame_idx_from_dataset(0),
    m_use_sphere_tracing(true),
    m_nr_iters_sphere_tracing(1),
    m_sphere_trace_agressiveness(1.0),
    m_sphere_trace_threshold_converged(0.001),
    m_chunk_size(2000),
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

    if (ImGui::CollapsingHeader("Render from Frame") ) {
        ImGui::Checkbox("use_controlable_frame", &m_use_controlable_frame);
        ImGui::Checkbox("m_use_sphere_tracing", &m_use_sphere_tracing);
        ImGui::SliderInt("nr_iters_sphere_tracing", &m_nr_iters_sphere_tracing, 1, 10) ;
        ImGui::SliderFloat("sphere_trace_agressiveness", &m_sphere_trace_agressiveness, 0.0, 3.0) ;
        ImGui::SliderFloat("sphere_trace_threshold_converged", &m_sphere_trace_threshold_converged, 0.0, 0.1) ;
        ImGui::SliderInt("chunk_size", &m_chunk_size, 0, 20000) ;
        ImGui::SliderInt("frame_idx_from_dataset", &m_frame_idx_from_dataset, 0, 100) ;
        ImGui::SliderFloat("isolines_layer_z_coord", &m_isolines_layer_z_coord, -1.0f, 1.0f) ;
        ImGui::SliderFloat("isoline_width", &m_isoline_width, 0.0f, 0.05f) ;
        ImGui::SliderFloat("distance_between_isolines", &m_distance_between_isolines, 0.0f, 0.3f) ;
        if (ImGui::Button("m_compute_full_layer") ) {
            m_compute_full_layer=true;
        }
        if (ImGui::Button("Render Full Image") ) {
            m_render_full_img=true;
        }
    }

    
    ImGui::End();

}

