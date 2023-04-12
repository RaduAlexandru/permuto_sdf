#pragma once


#include <torch/torch.h>
#include "permuto_sdf/helper_math.h"




#define BLOCK_SIZE 256






//intersects rays of shape Nx3 with a box. Return t value along the ray, and the positions themselves. If no hit then return 0
//return the points at the first intereseciton together with the T value along the ray, also the the 3D points at the exit of the box and the v value
//based on https://viclw17.github.io/2018/07/16/raytracing-ray-sphere-intersection/
//also https://www.programiz.com/cpp-programming/examples/quadratic-roots
__global__ void 
ray_intersection_gpu(
    const int nr_rays,
    const float sphere_radius,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> sphere_center,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_origins,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_dirs,
    //output
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_points_entry,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_t_entry,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_points_exit,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_t_exit,
    torch::PackedTensorAccessor32<bool,2,torch::RestrictPtrTraits> does_ray_intersect
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_rays){ //don't go out of bounds
        return;
    }

    float value_invalid_depth=0.0; 

    //load some stuff
    float3 ray_origin =make_float3(ray_origins[idx][0], ray_origins[idx][1], ray_origins[idx][2]);
    float3 ray_dir =make_float3(ray_dirs[idx][0], ray_dirs[idx][1], ray_dirs[idx][2]);
    float3 sphere_center_f3=make_float3(sphere_center[0], sphere_center[1], sphere_center[2]);


    float3 origin_centered=ray_origin-sphere_center_f3;
    float a = dot(ray_dir, ray_dir); //if the directions are normalized this is only ones so maybe it's not really needed
    float b = 2.0*dot(origin_centered, ray_dir);
    float c = dot(origin_centered,origin_centered) - sphere_radius*sphere_radius;
    float discriminant = b*b - 4*a*c;  //if the discriminant is negative then ther is no intersection 

    //first intersection and second intersection
    float t0= (-b - sqrtf( abs(discriminant) )) / (2.0*a); 
    float t1= (-b + sqrtf( abs(discriminant) )) / (2.0*a); 

    bool does_not_hit=discriminant < 0;
    bool is_hit=!does_not_hit;
    if(does_not_hit){
        t0=value_invalid_depth;
        t1=value_invalid_depth;
    }

    //t0 can end up being negative is the camera if inside the sphere so we clamp it to zero so that t0 is always at least in front of the camera
    //however making rays starting directly from the camera origin can lead to the regions just in front of the camera to be underconstrained if no other camera sees that region. This can lead to weird reconstruction where tiny images are created in front of every camera, theoretically driving the RGB loss to zero but not generalizing to novel views. Ideally the user would place the cameras so that not many rays are created in unconstrained regions or at least that the sphere_init is close enough to the object that we want to reconstruct.
    t0=max(0.0f, t0);

    float3 point_intersection_t0=ray_origin + t0*ray_dir;
    float3 point_intersection_t1=ray_origin + t1*ray_dir;

    //return
    //points entry into the sphere
    ray_points_entry[idx][0]=point_intersection_t0.x;
    ray_points_entry[idx][1]=point_intersection_t0.y;
    ray_points_entry[idx][2]=point_intersection_t0.z;
    //ray_t_entry
    ray_t_entry[idx][0]=t0;
    //ray_points_exit
    ray_points_exit[idx][0]=point_intersection_t1.x;
    ray_points_exit[idx][1]=point_intersection_t1.y;
    ray_points_exit[idx][2]=point_intersection_t1.z;
    //ray_t_exit
    ray_t_exit[idx][0]=t1;
    //does_ray_intersect
    does_ray_intersect[idx][0]=is_hit;
        
        


}

//based on https://stackoverflow.com/a/5408843
__global__ void 
rand_points_inside_gpu(
    const int nr_points,
    const float sphere_radius,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> sphere_center,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> phi_tensor,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> costheta_tensor,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> u_tensor,
    //output
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> points
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_points){ //don't go out of bounds
        return;
    }

    float phi=phi_tensor[idx];
    float costheta=costheta_tensor[idx];
    float u=u_tensor[idx];

    float theta=acos(costheta);
    float r=sphere_radius*pow(u, 1.0/3);

    float x= r * sin( theta) * cos( phi );
    float y = r * sin( theta) * sin( phi );
    float z = r * cos( theta );

    points[idx][0]=x;
    points[idx][1]=y;
    points[idx][2]=z;


}
