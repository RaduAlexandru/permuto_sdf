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
    int m_nr_samples_per_ray;
    float m_inv_s;
    float m_inv_s_min;
    float m_inv_s_max;
    float m_volsdf_beta;
    float m_neus_cos_anneal;
    float m_neus_variance;
    float m_nerf_surface_beta;
    float m_nerf_surface_std;
    float m_surface_prob_sigma;
    float m_surface_prob_height;
    float m_soft_opacity_sigma;
    float m_sphere_y_shift;
    bool m_show_unisurf_weights;
    bool m_show_volsdf_weights;
    bool m_show_neus_weights;
    bool m_show_nerf_surface_weights;
    float m_ray_origin_x_shift;
    float m_ray_origin_y_shift;


    //for the scripts for paper
    //visualiza isolines
    float m_isolines_layer_z_coord; //how far along the z coordinate of the camera should we place the layer
    bool m_compute_full_layer;
    float m_isoline_width;
    float m_distance_between_isolines;

    //when training the sdf_dense and hash we can choose to only visualzie the sdf from the dense grid
    bool m_use_only_dense_grid;

    //mitsuba
    int m_spp;
    bool m_render_mitsuba;
    int m_mitsuba_res_x;
    int m_mitsuba_res_y;


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

    //misc
    float m_error_map_max;

    


private:
    NGPGui(const std::shared_ptr<easy_pbr::Viewer>& view);

    std::shared_ptr<easy_pbr::Viewer> m_view;


    void install_callbacks(const std::shared_ptr<easy_pbr::Viewer>& view); //installs some callbacks that will be called by the viewer after it finishes an update

    //post draw callbacks
    void post_draw(easy_pbr::Viewer& view);


};