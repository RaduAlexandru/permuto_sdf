import torch 
import numpy as np

HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-6      # float32 only has 7 decimal digits precision

class AABB:
    #init with AABB([1.0, 1.0, 1.0], [0,0,0])
    def __init__(self, bounding_box_sizes_xyz, bounding_box_translation):
        self.bounding_box_sizes_xyz = bounding_box_sizes_xyz
        self.bounding_box_translation = bounding_box_translation
        self.invalid_depth_val=torch.tensor([[0.0]])

        #x
        xmin=-self.bounding_box_sizes_xyz[0]/2
        xmax=self.bounding_box_sizes_xyz[0]/2
        #y
        ymin=-self.bounding_box_sizes_xyz[1]/2
        ymax=self.bounding_box_sizes_xyz[1]/2
        #z
        zmin=-self.bounding_box_sizes_xyz[2]/2
        zmax=self.bounding_box_sizes_xyz[2]/2

        self.max_bounds=torch.tensor([xmax, ymax, zmax])
        self.min_bounds=torch.tensor([xmin, ymin, zmin])

    #given points of size Nx3 return a tensor of Nx1 if the points are inside the primitive
    def check_point_inside_primitive(self, points):
        # X_min <= X <= X_max and Y_min <= Y <= Y_max  and Z_min <= Z <= Z_max
        # points=points.view(-1,3)

        #put it at the origin of the box
        points=points-torch.as_tensor(self.bounding_box_translation)


        is_valid_max=points<self.max_bounds
        is_valid_min=points>self.min_bounds
        is_valid_min_max=torch.logical_and(is_valid_max, is_valid_min) #nx3 true if each coordinate is within the bound
        is_valid_points=  (is_valid_min_max*1.0).sum(dim=-1, keepdim=True) ==3 #set the points to true (valid) if all three coordinates are valid


        return is_valid_points

    #intersects rays of shape Nx3 with a box. Return t value along the ray, and the positions themselves. If no hit then return 0
    #based on https://github.com/stackgl/ray-aabb-intersection/blob/master/index.js
    #return the points at the first intereseciton together with the T value along the ray, also the the 3D points at the exit of the box and the v value
    def ray_intersection(self, ray_origins, ray_dirs):
        bb_min=   -torch.as_tensor(self.bounding_box_sizes_xyz)/2  +torch.as_tensor(self.bounding_box_translation)
        bb_max=  torch.as_tensor(self.bounding_box_sizes_xyz).cuda() -torch.as_tensor(self.bounding_box_sizes_xyz)/2  +torch.as_tensor(self.bounding_box_translation)

        dims=ray_dirs.shape[-1]

        lo= - torch.ones_like( ray_origins[:,0:1]) * HUGE_NUMBER
        hi=torch.ones_like( ray_origins[:,0:1]) * HUGE_NUMBER
        invalid_hit_all_dimensions=torch.zeros_like( ray_origins[:,0:1]).bool() #all false
        value_invalid_depth=self.invalid_depth_val



        for i in range(dims):
            # print ("is is ", i)
            dimLo = (bb_min[i] - ray_origins[:,i]) / ray_dirs[:,i]
            dimHi = (bb_max[i] - ray_origins[:,i]) / ray_dirs[:,i]
            dimLo=dimLo.unsqueeze(-1)
            dimHi=dimHi.unsqueeze(-1)

            

            #swap in order to make sure low is less than high
            low_bigger_than_hi=dimLo > dimHi
            dimLo_orignal=dimLo
            dimLo=torch.where( low_bigger_than_hi, dimHi, dimLo)
            dimHi=torch.where( low_bigger_than_hi, dimLo_orignal, dimHi)

            #set to invalid if we don't hit. If this dimension's high is less than the low we got then we definitely missed. http://youtu.be/USjbg5QXk3g?t=7m16s
            cond1=dimHi < lo
            cond2=dimLo > hi
            invalid_hit_this_dimension=torch.logical_or(cond1, cond2)
            invalid_hit_all_dimensions=torch.logical_or(invalid_hit_all_dimensions, invalid_hit_this_dimension) #set all dimensions to invalid if this one is invalid

            lo=torch.where( dimLo>lo, dimLo, lo)
            hi=torch.where(dimHi < hi, dimHi, hi)

        
        lo=torch.where( invalid_hit_all_dimensions, value_invalid_depth, lo)
        hi=torch.where( invalid_hit_all_dimensions, value_invalid_depth, hi)

        #lo can end up being negative if the camera is inside the bounding box so we clamp it to zero so that t0 is always at least in front of the camera
        #however making rays starting directly from the camera origin can lead to the regions just in front of the camera to be underconstrained if no other camera sees that region. This can lead to weird reconstruction where tiny images are created in front of every camera, theoretically driving the RGB loss to zero but not generalizing to novel views. Ideally the user would place the cameras so that not many rays are created in unconstrained regions or at least that the sphere_init is close enough to the object that we want to reconstruct.
        lo=torch.clamp(lo, min=0.0)


        #lo is the distance along the ray to the minimum intersection point
        lo_points=ray_origins+ lo*ray_dirs
        hi_points=ray_origins+ hi*ray_dirs

        does_ray_intersect_box=torch.logical_not(invalid_hit_all_dimensions)
        

        return lo_points, lo, hi_points, hi, does_ray_intersect_box 


    def remove_points_outside(self, mesh):
        points=torch.from_numpy(mesh.V).float().cuda()
        is_valid=self.check_point_inside_primitive(points)
        mesh.remove_marked_vertices( is_valid.flatten().bool().cpu().numpy() ,True)
        return mesh

    
    def rand_points_inside(self, nr_points):
        points=torch.rand(nr_points, 3) #in range 0,1
        points=points* torch.as_tensor(self.bounding_box_sizes_xyz).cuda() -torch.as_tensor(self.bounding_box_sizes_xyz)/2  +torch.as_tensor(self.bounding_box_translation)

        return points

    #given ray points and ray dirs, we shift the ray samples so they are inside the cube
    def cap_points_to_primitive_boundary(self, points, ray_origins, ray_dirs):
        t=(points-ray_origins).norm(dim=-1,keepdim=True)
        value_invalid_depth=torch.tensor([[0.0]])

        ray_points_entry, ray_t_entry, ray_points_exit, ray_t_exit, does_ray_intersect_box=self.ray_intersection(ray_origins, ray_dirs)

        #if the ray does not intersect the box, set the capped t to zero
        capped_t=torch.where( does_ray_intersect_box, t, value_invalid_depth)
        #if the t is larger than the exit one, then we set it to the exit
        capped_t=torch.where( capped_t>ray_t_exit, ray_t_exit, capped_t)
        #if the t is smaller than the entry, then we set it to the entry
        capped_t=torch.where( capped_t<ray_t_entry, ray_t_entry, capped_t)

        capped_points=ray_origins+capped_t*ray_dirs

        # is_point_capped=

        return capped_points, capped_t
