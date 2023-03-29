#pragma once


#include <torch/torch.h>
#include "permuto_sdf/helper_math.h"

// //Add this header after we add all cuda stuff because we need the profiler to have cudaDeviceSyncronize defined
// #define ENABLE_CUDA_PROFILING 1
// #include "Profiler.h" 

//matrices
#include "permuto_sdf/mat3.h"
#include "permuto_sdf/mat4.h"



#define BLOCK_SIZE 256






__global__ void 
random_rays_from_reel_gpu(
    const int nr_rays,
    const int nr_images,
    const int img_height,
    const int img_width,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> rgb_reel,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> mask_reel,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> K_reel,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> tf_world_cam_reel,
    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> pixel_indices,
    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> img_indices,
    const bool has_mask,
    //output
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_origins,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_dirs,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> gt_rgb,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> gt_mask
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_rays){ //don't go out of bounds
        return;
    }

    int img_idx=img_indices[idx];
    // img_idx=0;

    int pixel_1d=pixel_indices[idx];
    float3 point_2d_screen_coord;
    point_2d_screen_coord.x=pixel_1d%img_width;
    point_2d_screen_coord.y=pixel_1d/img_width;
    point_2d_screen_coord.z=1.0;
    //shift by 0.5 so we are in the center of the pixel
    point_2d_screen_coord.x+=0.5;
    point_2d_screen_coord.y+=0.5;



    //get to cam coord by multiplying with the K inverse
    float fx=K_reel[img_idx][0][0];
    float fy=K_reel[img_idx][1][1];
    float cx=K_reel[img_idx][0][2];
    float cy=K_reel[img_idx][1][2];
    float3 point_cam_coord; 
    point_cam_coord.x= (point_2d_screen_coord.x - cx )/fx;
    point_cam_coord.y= (point_2d_screen_coord.y - cy )/fy;
    point_cam_coord.z=1.0;



    //get from cam coords to world coordinate
    mat3 R(
            tf_world_cam_reel[img_idx][0][0], tf_world_cam_reel[img_idx][1][0], tf_world_cam_reel[img_idx][2][0],
            tf_world_cam_reel[img_idx][0][1], tf_world_cam_reel[img_idx][1][1], tf_world_cam_reel[img_idx][2][1],
            tf_world_cam_reel[img_idx][0][2], tf_world_cam_reel[img_idx][1][2], tf_world_cam_reel[img_idx][2][2]
             ); // needs to be added in column major order
    float3 t;
    t.x=tf_world_cam_reel[img_idx][0][3];
    t.y=tf_world_cam_reel[img_idx][1][3];
    t.z=tf_world_cam_reel[img_idx][2][3];

    float3 pixel_world_coord=R*point_cam_coord;
    pixel_world_coord+=t;

    //get ray dir and origin
    float3 ray_dir=normalize(pixel_world_coord-t);
    float3 ray_origin=t;


    //get also the gt_rgb pixel
    int x=floor(point_2d_screen_coord.x);
    int y=floor(point_2d_screen_coord.y);
    float3 gt_rgb_pixel;
    gt_rgb_pixel.x=rgb_reel[img_idx][0][y][x];
    gt_rgb_pixel.y=rgb_reel[img_idx][1][y][x];
    gt_rgb_pixel.z=rgb_reel[img_idx][2][y][x];
    //get mask
    float mask_pixel=1.0;
    if(has_mask){
        mask_pixel=mask_reel[img_idx][0][y][x];
    }



    //write everything to output
    //origin
    ray_origins[idx][0]=ray_origin.x;
    ray_origins[idx][1]=ray_origin.y;
    ray_origins[idx][2]=ray_origin.z;
    //dir
    ray_dirs[idx][0]=ray_dir.x;
    ray_dirs[idx][1]=ray_dir.y;
    ray_dirs[idx][2]=ray_dir.z;
    //gtrgb
    gt_rgb[idx][0]=gt_rgb_pixel.x*mask_pixel;
    gt_rgb[idx][1]=gt_rgb_pixel.y*mask_pixel;
    gt_rgb[idx][2]=gt_rgb_pixel.z*mask_pixel;
    //mask
    gt_mask[idx][0]=mask_pixel;


}


__global__ void 
rays_from_reprojection_reel_gpu(
    const int nr_points,
    const int nr_images,
    const int img_height,
    const int img_width,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> rgb_reel,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> mask_reel,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> K_reel,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> tf_cam_world_reel,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> tf_world_cam_reel,
    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> pixel_indices,
    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> img_indices,
    const bool has_mask,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> points_reprojected,
    //output
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_origins,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> ray_dirs,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> gt_rgb,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> gt_mask
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_points){ //don't go out of bounds
        return;
    }



    //get the points in 3d 
    float3 point_world_coord=make_float3(points_reprojected[idx][0], points_reprojected[idx][1], points_reprojected[idx][2]);


    //get a random img towards which we project onto
    int img_idx=img_indices[idx];
    //get the transofmr from world to this cam
    mat3 R_cam_world(
            tf_cam_world_reel[img_idx][0][0], tf_cam_world_reel[img_idx][1][0], tf_cam_world_reel[img_idx][2][0],
            tf_cam_world_reel[img_idx][0][1], tf_cam_world_reel[img_idx][1][1], tf_cam_world_reel[img_idx][2][1],
            tf_cam_world_reel[img_idx][0][2], tf_cam_world_reel[img_idx][1][2], tf_cam_world_reel[img_idx][2][2]
             ); // needs to be added in column major order
    float3 t_cam_world;
    t_cam_world.x=tf_cam_world_reel[img_idx][0][3];
    t_cam_world.y=tf_cam_world_reel[img_idx][1][3];
    t_cam_world.z=tf_cam_world_reel[img_idx][2][3];

    float3 pixel_cam_coord=R_cam_world*point_world_coord;
    pixel_cam_coord+=t_cam_world;

    //get to K and transform to 2D
    float fx=K_reel[img_idx][0][0];
    float fy=K_reel[img_idx][1][1];
    float cx=K_reel[img_idx][0][2];
    float cy=K_reel[img_idx][1][2];
    float3 point_2d_screen_coord; 
    point_2d_screen_coord.x= pixel_cam_coord.x *fx/pixel_cam_coord.z + cx;
    point_2d_screen_coord.y= pixel_cam_coord.y *fy/pixel_cam_coord.z + cy;
    

    //get the pixel that we are getting from here
    bool is_within_bounds=true;
    if(floor(point_2d_screen_coord.x)<0 || floor(point_2d_screen_coord.x)>=img_width || floor(point_2d_screen_coord.y)<0 || floor(point_2d_screen_coord.y)>=img_height){
        is_within_bounds=false;
    }
    //if we are out of bounds, we get a random pixel
    if(!is_within_bounds){
        int pixel_1d=pixel_indices[idx];
        point_2d_screen_coord.x=pixel_1d%img_width;
        point_2d_screen_coord.y=pixel_1d/img_width;
        point_2d_screen_coord.z=1.0;
        //shift by 0.5 so we are in the center of the pixel
        point_2d_screen_coord.x+=0.5;
        point_2d_screen_coord.y+=0.5;
    }

    //now given this 2D point, we get the ray_direction of it by reprojecting it to 3D again, we have to do this because since we can be out of bounds, the 3D point will not correspond to the one given
    //-----------------------------------------------------
    //get to cam coord by multiplying with the K inverse
    float3 point_cam_coord;
    point_cam_coord.x= (point_2d_screen_coord.x - cx )/fx;
    point_cam_coord.y= (point_2d_screen_coord.y - cy )/fy;
    point_cam_coord.z=1.0;
    //get from cam coords to world coordinate
    mat3 R_world_cam(
            tf_world_cam_reel[img_idx][0][0], tf_world_cam_reel[img_idx][1][0], tf_world_cam_reel[img_idx][2][0],
            tf_world_cam_reel[img_idx][0][1], tf_world_cam_reel[img_idx][1][1], tf_world_cam_reel[img_idx][2][1],
            tf_world_cam_reel[img_idx][0][2], tf_world_cam_reel[img_idx][1][2], tf_world_cam_reel[img_idx][2][2]
             ); // needs to be added in column major order
    float3 t_world_cam;
    t_world_cam.x=tf_world_cam_reel[img_idx][0][3];
    t_world_cam.y=tf_world_cam_reel[img_idx][1][3];
    t_world_cam.z=tf_world_cam_reel[img_idx][2][3];

    float3 pixel_world_coord=R_world_cam*point_cam_coord;
    pixel_world_coord+=t_world_cam;

    



    //get ray dir and origin
    //get ray dir and origin
    float3 ray_dir=normalize(pixel_world_coord-t_world_cam);
    float3 ray_origin=t_world_cam;


    //get also the gt_rgb pixel
    int x=floor(point_2d_screen_coord.x);
    int y=floor(point_2d_screen_coord.y);
    float3 gt_rgb_pixel;
    gt_rgb_pixel.x=rgb_reel[img_idx][0][y][x];
    gt_rgb_pixel.y=rgb_reel[img_idx][1][y][x];
    gt_rgb_pixel.z=rgb_reel[img_idx][2][y][x];
    //get mask
    float mask_pixel=1.0;
    if(has_mask){
        mask_pixel=mask_reel[img_idx][0][y][x];
    }



    //write everything to output
    //origin
    ray_origins[idx][0]=ray_origin.x;
    ray_origins[idx][1]=ray_origin.y;
    ray_origins[idx][2]=ray_origin.z;
    //dir
    ray_dirs[idx][0]=ray_dir.x;
    ray_dirs[idx][1]=ray_dir.y;
    ray_dirs[idx][2]=ray_dir.z;
    //gtrgb
    gt_rgb[idx][0]=gt_rgb_pixel.x*mask_pixel;
    gt_rgb[idx][1]=gt_rgb_pixel.y*mask_pixel;
    gt_rgb[idx][2]=gt_rgb_pixel.z*mask_pixel;
    //mask
    gt_mask[idx][0]=mask_pixel;



}




__global__ void 
spherical_harmonics_gpu(
    const int nr_elements,
    const int degree,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> dirs,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> out
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value


    if(idx>=nr_elements){ //don't go out of bounds
        return;
    }


    float x = dirs[idx][0];
	float y = dirs[idx][1];
	float z = dirs[idx][2];


    // Let compiler figure out how to sequence/reorder these calculations w.r.t. branches
	float xy=x*y, xz=x*z, yz=y*z, x2=x*x, y2=y*y, z2=z*z;
	float x4=x2*x2, y4=y2*y2, z4=z2*z2;
	float x6=x4*x2, y6=y4*y2, z6=z4*z2;

    // SH polynomials generated using scripts/gen_sh.py based on the recurrence relations in appendix A1 of https://www.ppsloan.org/publications/StupidSH36.pdf


    out[idx][0] = 0.28209479177387814f;                          // 1/(2*sqrt(pi))
    if (degree <= 1) { return; }
    out[idx][1] = -0.48860251190291987f*y;                               // -sqrt(3)*y/(2*sqrt(pi))
    out[idx][2] = 0.48860251190291987f*z;                                // sqrt(3)*z/(2*sqrt(pi))
    out[idx][3]  = -0.48860251190291987f*x;                               // -sqrt(3)*x/(2*sqrt(pi))
    if (degree <= 2) { return; }
    out[idx][4]  = 1.0925484305920792f*xy;                                // sqrt(15)*xy/(2*sqrt(pi))
    out[idx][5]  = -1.0925484305920792f*yz;                               // -sqrt(15)*yz/(2*sqrt(pi))
    out[idx][6]  = 0.94617469575755997f*z2 - 0.31539156525251999f;                         // sqrt(5)*(3*z2 - 1)/(4*sqrt(pi))
    out[idx][7]  = -1.0925484305920792f*xz;                               // -sqrt(15)*xz/(2*sqrt(pi))
    out[idx][8]  = 0.54627421529603959f*x2 - 0.54627421529603959f*y2;                              // sqrt(15)*(x2 - y2)/(4*sqrt(pi))
    if (degree <= 3) { return; }
    out[idx][9]  = 0.59004358992664352f*y*(-3.0f*x2 + y2);                         // sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
    out[idx][10]  = 2.8906114426405538f*xy*z;                             // sqrt(105)*xy*z/(2*sqrt(pi))
    out[idx][11]  = 0.45704579946446572f*y*(1.0f - 5.0f*z2);                                // sqrt(42)*y*(1 - 5*z2)/(8*sqrt(pi))
    out[idx][12]  = 0.3731763325901154f*z*(5.0f*z2 - 3.0f);                         // sqrt(7)*z*(5*z2 - 3)/(4*sqrt(pi))
    out[idx][13]  = 0.45704579946446572f*x*(1.0f - 5.0f*z2);                                // sqrt(42)*x*(1 - 5*z2)/(8*sqrt(pi))
    out[idx][14]  = 1.4453057213202769f*z*(x2 - y2);                              // sqrt(105)*z*(x2 - y2)/(4*sqrt(pi))
    out[idx][15]  = 0.59004358992664352f*x*(-x2 + 3.0f*y2);                                // sqrt(70)*x*(-x2 + 3*y2)/(8*sqrt(pi))
    if (degree <= 4) { return; }
    out[idx][16]  = 2.5033429417967046f*xy*(x2 - y2);                             // 3*sqrt(35)*xy*(x2 - y2)/(4*sqrt(pi))
    out[idx][17]  = 1.7701307697799304f*yz*(-3.0f*x2 + y2);                                // 3*sqrt(70)*yz*(-3*x2 + y2)/(8*sqrt(pi))
    out[idx][18]  = 0.94617469575756008f*xy*(7.0f*z2 - 1.0f);                               // 3*sqrt(5)*xy*(7*z2 - 1)/(4*sqrt(pi))
    out[idx][19]  = 0.66904654355728921f*yz*(3.0f - 7.0f*z2);                               // 3*sqrt(10)*yz*(3 - 7*z2)/(8*sqrt(pi))
    out[idx][20]  = -3.1735664074561294f*z2 + 3.7024941420321507f*z4 + 0.31735664074561293f;                                // 3*(-30*z2 + 35*z4 + 3)/(16*sqrt(pi))
    out[idx][21]  = 0.66904654355728921f*xz*(3.0f - 7.0f*z2);                               // 3*sqrt(10)*xz*(3 - 7*z2)/(8*sqrt(pi))
    out[idx][22]  = 0.47308734787878004f*(x2 - y2)*(7.0f*z2 - 1.0f);                                // 3*sqrt(5)*(x2 - y2)*(7*z2 - 1)/(8*sqrt(pi))
    out[idx][23]  = 1.7701307697799304f*xz*(-x2 + 3.0f*y2);                                // 3*sqrt(70)*xz*(-x2 + 3*y2)/(8*sqrt(pi))
    out[idx][24]  = -3.7550144126950569f*x2*y2 + 0.62583573544917614f*x4 + 0.62583573544917614f*y4;                         // 3*sqrt(35)*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
    if (degree <= 5) { return; }
    out[idx][25]  = 0.65638205684017015f*y*(10.0f*x2*y2 - 5.0f*x4 - y4);                            // 3*sqrt(154)*y*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
    out[idx][26]  = 8.3026492595241645f*xy*z*(x2 - y2);                           // 3*sqrt(385)*xy*z*(x2 - y2)/(4*sqrt(pi))
    out[idx][27]  = -0.48923829943525038f*y*(3.0f*x2 - y2)*(9.0f*z2 - 1.0f);                         // -sqrt(770)*y*(3*x2 - y2)*(9*z2 - 1)/(32*sqrt(pi))
    out[idx][28]  = 4.7935367849733241f*xy*z*(3.0f*z2 - 1.0f);                              // sqrt(1155)*xy*z*(3*z2 - 1)/(4*sqrt(pi))
    out[idx][29]  = 0.45294665119569694f*y*(14.0f*z2 - 21.0f*z4 - 1.0f);                             // sqrt(165)*y*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
    out[idx][30]  = 0.1169503224534236f*z*(-70.0f*z2 + 63.0f*z4 + 15.0f);                            // sqrt(11)*z*(-70*z2 + 63*z4 + 15)/(16*sqrt(pi))
    out[idx][31]  = 0.45294665119569694f*x*(14.0f*z2 - 21.0f*z4 - 1.0f);                             // sqrt(165)*x*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
    out[idx][32]  = 2.3967683924866621f*z*(x2 - y2)*(3.0f*z2 - 1.0f);                               // sqrt(1155)*z*(x2 - y2)*(3*z2 - 1)/(8*sqrt(pi))
    out[idx][33]  = -0.48923829943525038f*x*(x2 - 3.0f*y2)*(9.0f*z2 - 1.0f);                         // -sqrt(770)*x*(x2 - 3*y2)*(9*z2 - 1)/(32*sqrt(pi))
    out[idx][34]  = 2.0756623148810411f*z*(-6.0f*x2*y2 + x4 + y4);                         // 3*sqrt(385)*z*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
    out[idx][35]  = 0.65638205684017015f*x*(10.0f*x2*y2 - x4 - 5.0f*y4);                            // 3*sqrt(154)*x*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
    if (degree <= 6) { return; }
    out[idx][36] = 1.3663682103838286f*xy*(-10.0f*x2*y2 + 3.0f*x4 + 3.0f*y4);                               // sqrt(6006)*xy*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
    out[idx][37] = 2.3666191622317521f*yz*(10.0f*x2*y2 - 5.0f*x4 - y4);                            // 3*sqrt(2002)*yz*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
    out[idx][38] = 2.0182596029148963f*xy*(x2 - y2)*(11.0f*z2 - 1.0f);                             // 3*sqrt(91)*xy*(x2 - y2)*(11*z2 - 1)/(8*sqrt(pi))
    out[idx][39] = -0.92120525951492349f*yz*(3.0f*x2 - y2)*(11.0f*z2 - 3.0f);                               // -sqrt(2730)*yz*(3*x2 - y2)*(11*z2 - 3)/(32*sqrt(pi))
    out[idx][40] = 0.92120525951492349f*xy*(-18.0f*z2 + 33.0f*z4 + 1.0f);                           // sqrt(2730)*xy*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
    out[idx][41] = 0.58262136251873131f*yz*(30.0f*z2 - 33.0f*z4 - 5.0f);                            // sqrt(273)*yz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
    out[idx][42] = 6.6747662381009842f*z2 - 20.024298714302954f*z4 + 14.684485723822165f*z6 - 0.31784601133814211f;                         // sqrt(13)*(105*z2 - 315*z4 + 231*z6 - 5)/(32*sqrt(pi))
    out[idx][43] = 0.58262136251873131f*xz*(30.0f*z2 - 33.0f*z4 - 5.0f);                            // sqrt(273)*xz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
    out[idx][44] = 0.46060262975746175f*(x2 - y2)*(11.0f*z2*(3.0f*z2 - 1.0f) - 7.0f*z2 + 1.0f);                               // sqrt(2730)*(x2 - y2)*(11*z2*(3*z2 - 1) - 7*z2 + 1)/(64*sqrt(pi))
    out[idx][45] = -0.92120525951492349f*xz*(x2 - 3.0f*y2)*(11.0f*z2 - 3.0f);                               // -sqrt(2730)*xz*(x2 - 3*y2)*(11*z2 - 3)/(32*sqrt(pi))
    out[idx][46] = 0.50456490072872406f*(11.0f*z2 - 1.0f)*(-6.0f*x2*y2 + x4 + y4);                          // 3*sqrt(91)*(11*z2 - 1)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
    out[idx][47] = 2.3666191622317521f*xz*(10.0f*x2*y2 - x4 - 5.0f*y4);                            // 3*sqrt(2002)*xz*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
    out[idx][48] = 10.247761577878714f*x2*y4 - 10.247761577878714f*x4*y2 + 0.6831841051919143f*x6 - 0.6831841051919143f*y6;                         // sqrt(6006)*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
    if (degree <= 7) { return; }


    


}


__global__ void 
update_errors_of_matching_indices_gpu(
    const int nr_old_errors,
    const int nr_new_errors,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> old_indices,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> old_errors,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> new_indices,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> new_errors,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> updated_errors
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value


    if(idx>=nr_old_errors){ //don't go out of bounds
        return;
    }

    long cur_old_idx=old_indices[idx];
    float cur_old_error=old_errors[idx];

    updated_errors[idx]=old_errors[idx];

    if (cur_old_idx!=0){
        //check of we have an idx that is the same in new_indices, and if it matches then we just update
        for(int i=0; i<nr_new_errors; i++){
            long cur_new_idx=new_indices[i];
            float cur_new_error=new_errors[i];
            //update the error because we sampled it in this frame
            if (cur_new_idx==cur_old_idx && cur_new_idx!=0){
                updated_errors[idx]=new_errors[idx];
            }
            
        }
    }


    


}



__device__ float clamp_gpu(float x, float a, float b){
  return max(a, min(b, x));
}

__device__ float map_range_gpu(const float input, const float input_start,const float input_end, const float output_start, const float output_end) {
    //we clamp the input between the start and the end
    float input_clamped=clamp_gpu(input, input_start, input_end);
    return output_start + ((output_end - output_start) / (input_end - input_start)) * (input_clamped - input_start);
}

__global__ void 
meshgrid3d_gpu(
    const int nr_points_per_dim,
    const float min, 
    const float max,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> grid
    ) {

    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int z = blockIdx.z * blockDim.z + threadIdx.z;


    if(x>=nr_points_per_dim){ return; } //don't go out of bounds 
    if(y>=nr_points_per_dim){ return; } //don't go out of bounds 
    if(z>=nr_points_per_dim){ return; } //don't go out of bounds 
    // printf("x,y,z is %d, %d, %d \n",x,y,z); 


    float x_value=map_range_gpu(x, 0, nr_points_per_dim-1, min, max);
    float y_value=map_range_gpu(y, 0, nr_points_per_dim-1, min, max);
    float z_value=map_range_gpu(z, 0, nr_points_per_dim-1, min, max); 


    grid[x][y][z][0]=x_value;
    grid[x][y][z][1]=y_value;
    grid[x][y][z][2]=z_value; 


}















