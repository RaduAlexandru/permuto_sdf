#pragma once



#include <memory>



namespace easy_pbr{
    class Mesh;
    class Viewer;
}


class NGPGui : public std::enable_shared_from_this<NGPGui>{
public:
    template <class ...Args>
    static std::shared_ptr<NGPGui> create( Args&& ...args ){
        return std::shared_ptr<NGPGui>( new NGPGui(std::forward<Args>(args)...) );
    }
    ~NGPGui();

    bool m_do_training;
    bool m_control_view;
    float m_time_val;
    float m_c2f_progress;


    //for the scripts for paper
    //visualiza isolines
    float m_isolines_layer_z_coord; //how far along the z coordinate of the camera should we place the layer
    bool m_compute_full_layer;
    float m_isoline_width;
    float m_distance_between_isolines;


    //stuff for rendeirng from arbitrary frames
    bool m_render_full_img;
    bool m_use_controlable_frame;
    int m_frame_idx_from_dataset;
    bool m_use_sphere_tracing;
    int m_nr_iters_sphere_tracing;
    float m_sphere_trace_agressiveness;
    float m_sphere_trace_threshold_converged;
    float m_sphere_trace_push_in_gradient_dir;
    int m_chunk_size;


    


private:
    NGPGui(const std::shared_ptr<easy_pbr::Viewer>& view);

    std::shared_ptr<easy_pbr::Viewer> m_view;


    void install_callbacks(const std::shared_ptr<easy_pbr::Viewer>& view); //installs some callbacks that will be called by the viewer after it finishes an update

    //post draw callbacks
    void post_draw(easy_pbr::Viewer& view);


};