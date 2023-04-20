#include "permuto_sdf/Sphere.cuh"

//c++

#include "UtilsPytorch.h" //contains torch so it has to be added BEFORE any other include because the other ones might include loguru which gets screwed up if torch was included before it

//my stuff
#include "permuto_sdf/SphereGPU.cuh"

//loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp> //needs to be added after torch.h otherwise loguru stops printing for some reason




using torch::Tensor;
using namespace radu::utils;


template <typename T>
T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}


//CPU code that calls the kernels
Sphere::Sphere(float radius, Eigen::Vector3f center):
    m_radius(radius),
    m_center(center)
    {

    m_center_tensor=eigen2tensor(center).squeeze(1).cuda();      //maek it size 3 only 

}

Sphere::~Sphere(){
}



std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
    Sphere::ray_intersection(const torch::Tensor& ray_origins, const torch::Tensor& ray_dirs){

    CHECK(ray_origins.dim()==2) << "ray_origins should have dim 2 correspondin to nr_raysx3. However it has sizes" << ray_origins.sizes();
    CHECK(ray_dirs.dim()==2) << "ray_dirs should have dim 2 correspondin to nr_raysx3. However it has sizes" << ray_dirs.sizes();


    int nr_rays=ray_origins.size(0);

    //return ray_points_entry, ray_t_entry, ray_points_exit, ray_t_exit, does_ray_intersect_box
    Tensor ray_points_entry=torch::empty({ nr_rays, 3 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    Tensor ray_t_entry=torch::empty({ nr_rays, 1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    Tensor ray_points_exit=torch::empty({ nr_rays, 3 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    Tensor ray_t_exit=torch::empty({ nr_rays, 1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    Tensor does_ray_intersect=torch::empty({ nr_rays, 1 },  torch::dtype(torch::kBool).device(torch::kCUDA, 0)  );



    const dim3 blocks = { (unsigned int)div_round_up(nr_rays, BLOCK_SIZE), 1, 1 };


    ray_intersection_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_rays,
        m_radius,
        m_center_tensor.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        ray_origins.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        ray_dirs.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        //output
        ray_points_entry.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        ray_t_entry.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        ray_points_exit.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        ray_t_exit.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        does_ray_intersect.packed_accessor32<bool,2,torch::RestrictPtrTraits>()
    );

    return std::make_tuple(ray_points_entry, ray_t_entry, ray_points_exit, ray_t_exit, does_ray_intersect);

}

//based on #based on https://stackoverflow.com/a/5408843
torch::Tensor Sphere::rand_points_inside(const int nr_points){


    Tensor phi_tensor = torch::empty({ nr_points },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  ).uniform_(0, 2*M_PI);
    Tensor costheta_tensor = torch::empty({ nr_points },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  ).uniform_(-1, 1);
    Tensor u_tensor = torch::rand({ nr_points },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );

    Tensor points=torch::empty({ nr_points, 3 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );


    const dim3 blocks = { (unsigned int)div_round_up(nr_points, BLOCK_SIZE), 1, 1 };


    rand_points_inside_gpu<<<blocks, BLOCK_SIZE>>>(
        nr_points,
        m_radius,
        m_center_tensor.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        phi_tensor.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        costheta_tensor.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        u_tensor.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        //output
        points.packed_accessor32<float,2,torch::RestrictPtrTraits>()
    );


    return points;

}

torch::Tensor Sphere::check_point_inside_primitive(const torch::Tensor& points){

    torch::Tensor points_in_sphere_coords=points-m_center_tensor.view({1,3});
    torch::Tensor point_dist_from_center_of_sphere=points_in_sphere_coords.norm(2, 1, true); // l2norm, dim ,keepdim

    torch::Tensor is_inside_primitive=point_dist_from_center_of_sphere<m_radius;

    return is_inside_primitive;
}

