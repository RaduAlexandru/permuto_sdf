#pragma once

#include <memory>
#include <stdarg.h>

#include <cuda.h>


#include "torch/torch.h"

#include <Eigen/Dense>

#ifdef HSDF_WITH_GL
    #include "easy_pbr/Frame.h"
    #include "easy_gl/Shader.h"
    #include "easy_gl/GBuffer.h"
#endif

#include "data_loaders/TensorReel.h"


namespace easy_pbr{
    class Mesh;
    class MeshGL;
    class Viewer;
}
namespace radu { namespace utils{
    class RandGenerator;
}}





class PermutoSDF : public std::enable_shared_from_this<PermutoSDF>{
public:
    template <class ...Args>
    static std::shared_ptr<PermutoSDF> create( Args&& ...args ){
        return std::shared_ptr<PermutoSDF>( new PermutoSDF(std::forward<Args>(args)...) );
    }
    ~PermutoSDF();


    
    //static stuff
    static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
        random_rays_from_reel(const TensorReel& reel, const int nr_rays);
    static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
        rays_from_reprojection_reel(const TensorReel& reel, const torch::Tensor& points_reprojected);
    static torch::Tensor spherical_harmonics(const torch::Tensor& dirs, const int degree);
    static torch::Tensor update_errors_of_matching_indices(const torch::Tensor& old_indices, const torch::Tensor& old_errors, const torch::Tensor& new_indices, const torch::Tensor& new_errors);
    static torch::Tensor meshgrid3d(const float min, const float max, const int nr_points_per_dim);
    //for sampling with low discrepancy
    static double phi(const unsigned &i);
    static Eigen::VectorXi low_discrepancy2d_sampling(const int nr_samples, const int height, const int width);

    
    #ifdef HSDF_WITH_GL
        void init_opengl();
        void compile_shaders();
        torch::Tensor render_atributes(const std::shared_ptr<easy_pbr::Mesh>& mesh, const easy_pbr::Frame frame);

        //render into uvt
        gl::GBuffer m_atrib_gbuffer; //we render into it xyz,dir,uv
        gl::Shader m_atrib_shader; //onyl used to render into the depth map and nothing else
    #endif

    std::shared_ptr<easy_pbr::Viewer> m_view;


    std::shared_ptr<radu::utils::RandGenerator> m_rand_gen;
    static std::shared_ptr<radu::utils::RandGenerator> m_rand_gen_static;



   

private:
    PermutoSDF( const std::shared_ptr<easy_pbr::Viewer>& view);

    static std::vector<unsigned char> lutLDBN_BNOT;
    static std::vector<unsigned char> lutLDBN_STEP;
    static std::vector<unsigned> mirror;

  
};


