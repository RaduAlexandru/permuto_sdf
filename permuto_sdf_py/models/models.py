import torch


import numpy as np
import os
import sys



from permuto_sdf_py.volume_rendering.volume_rendering_modules import *
from permuto_sdf_py.models.modules import CreateRaysModule
from permuto_sdf_py.utils.common_utils import map_range_val
from permuto_sdf_py.utils.common_utils import leaky_relu_init
from permuto_sdf_py.utils.common_utils import apply_weight_init_fn
from permuto_sdf import PermutoSDF
from permuto_sdf_py.utils.common_utils import TIME_START
from permuto_sdf_py.utils.common_utils import TIME_END

import permutohedral_encoding as permuto_enc


class MLP(torch.jit.ScriptModule):

    def __init__(self, in_channels, hidden_dim, out_channels, nr_layers, last_layer_linear_init):
        super(MLP, self).__init__()



        self.layers=[]
        self.layers.append(  torch.nn.Linear(in_channels, hidden_dim) )
        self.layers.append(  torch.nn.GELU() )
        for i in range(nr_layers):
            self.layers.append(   torch.nn.Linear(hidden_dim, hidden_dim)  )
            self.layers.append(   torch.nn.GELU()  )
        self.layers.append(  torch.nn.Linear(hidden_dim, out_channels  )   )

        self.mlp=torch.nn.Sequential(*self.layers)

        apply_weight_init_fn(self.mlp, leaky_relu_init, negative_slope=0.0)
        if last_layer_linear_init:
            leaky_relu_init(self.mlp[-1], negative_slope=1.0)

        self.weights_initialized=True #so that apply_weight_init_fn doesnt initialize anything

    @torch.jit.script_method
    def forward(self, x):

        x=self.mlp(x)

        return x

#from https://arxiv.org/pdf/2202.08345.pdf
class LipshitzMLP(torch.nn.Module):

    def __init__(self, in_channels, nr_out_channels_per_layer, last_layer_linear):
        super(LipshitzMLP, self).__init__()


        self.last_layer_linear=last_layer_linear
     

        self.layers=torch.nn.ModuleList()
        # self.layers=[]
        for i in range(len(nr_out_channels_per_layer)):
            cur_out_channels=nr_out_channels_per_layer[i]
            self.layers.append(  torch.nn.Linear(in_channels, cur_out_channels)   )
            in_channels=cur_out_channels
        apply_weight_init_fn(self, leaky_relu_init, negative_slope=0.0)
        if last_layer_linear:
            leaky_relu_init(self.layers[-1], negative_slope=1.0)

        #we make each weight separately because we want to add the normalize to it
        self.weights_per_layer=torch.nn.ParameterList()
        self.biases_per_layer=torch.nn.ParameterList()
        for i in range(len(self.layers)):
            self.weights_per_layer.append( self.layers[i].weight  )
            self.biases_per_layer.append( self.layers[i].bias  )

        self.lipshitz_bound_per_layer=torch.nn.ParameterList()
        for i in range(len(self.layers)):
            max_w= torch.max(torch.sum(torch.abs(self.weights_per_layer[i]), dim=1))
            #we actually make the initial value quite large because we don't want at the beggining to hinder the rgb model in any way. A large c means that the scale will be 1
            c = torch.nn.Parameter(  torch.ones((1))*max_w*2 ) 
            self.lipshitz_bound_per_layer.append(c)






        self.weights_initialized=True #so that apply_weight_init_fn doesnt initialize anything

    def normalization(self, w, softplus_ci):
        absrowsum = torch.sum(torch.abs(w), dim=1)
        # scale = torch.minimum(torch.tensor(1.0), softplus_ci/absrowsum)
        # this is faster than the previous line since we don't constantly recreate a torch.tensor(1.0)
        scale = softplus_ci/absrowsum
        scale = torch.clamp(scale, max=1.0)
        return w * scale[:,None]

    def lipshitz_bound_full(self):
        lipshitz_full=1
        for i in range(len(self.layers)):
            lipshitz_full=lipshitz_full*torch.nn.functional.softplus(self.lipshitz_bound_per_layer[i])

        return lipshitz_full

    def forward(self, x):

        # x=self.mlp(x)

        for i in range(len(self.layers)):
            weight=self.weights_per_layer[i]
            bias=self.biases_per_layer[i]

            weight=self.normalization(weight, torch.nn.functional.softplus(self.lipshitz_bound_per_layer[i])  )

            x=torch.nn.functional.linear(x, weight, bias)

            is_last_layer=i==(len(self.layers)-1)

            if is_last_layer and self.last_layer_linear:
                pass
            else:
                x=torch.nn.functional.gelu(x)


        return x

class SDF(torch.nn.Module):

    def __init__(self, in_channels, boundary_primitive, geom_feat_size_out, nr_iters_for_c2f):
        super(SDF, self).__init__()

        self.in_channels=in_channels
        self.boundary_primitive=boundary_primitive
        self.geom_feat_size_out=geom_feat_size_out


        #create encoding
        pos_dim=in_channels
        capacity=pow(2,18) #2pow18
        nr_levels=24 
        nr_feat_per_level=2 
        coarsest_scale=1.0 
        finest_scale=0.0001 
        scale_list=np.geomspace(coarsest_scale, finest_scale, num=nr_levels)
        self.encoding=permuto_enc.PermutoEncoding(pos_dim, capacity, nr_levels, nr_feat_per_level, scale_list, appply_random_shift_per_level=True, concat_points=True, concat_points_scaling=1e-3)           

        
        self.sdf_shift=1e-2
        self.mlp_sdf= torch.nn.Sequential(
            torch.nn.Linear(self.encoding.output_dims() ,32),
            torch.nn.GELU(),
            torch.nn.Linear(32,32),
            torch.nn.GELU(),
            torch.nn.Linear(32,32),
            torch.nn.GELU(),
            torch.nn.Linear(32,1+geom_feat_size_out)
        )
        apply_weight_init_fn(self.mlp_sdf, leaky_relu_init, negative_slope=0.0)
        leaky_relu_init(self.mlp_sdf[-1], negative_slope=1.0)
        with torch.set_grad_enabled(False):
            self.mlp_sdf[-1].bias+=self.sdf_shift #faster if we just put it in the bias

        # self.mlp_sdf=torch.compile(self.mlp_sdf, mode="max-autotune")

       


        self.c2f=permuto_enc.Coarse2Fine(nr_levels)
        self.nr_iters_for_c2f=nr_iters_for_c2f
        self.last_iter_nr=sys.maxsize

    def forward(self, points, iter_nr):

        assert points.shape[1] == self.in_channels, "points should be N x in_channels"

        self.last_iter_nr=iter_nr

       
        window=self.c2f( map_range_val(iter_nr, 0.0, self.nr_iters_for_c2f, 0.3, 1.0   ) )

     
        point_features=self.encoding(points, window.view(-1))
        sdf_and_feat=self.mlp_sdf(point_features)
        
        if self.geom_feat_size_out!=0:
            sdf=sdf_and_feat[:,0:1]
            geom_feat=sdf_and_feat[:,-self.geom_feat_size_out:]
        else:
            sdf=sdf_and_feat
            geom_feat=None


        return sdf, geom_feat

    def get_sdf_and_gradient(self, points, iter_nr, method="autograd"):


        if method=="finite_difference":
            with torch.set_grad_enabled(False):
                #to the original positions, add also a tiny epsilon in all directions
                nr_points_original=points.shape[0]
                epsilon=1e-4
                points_xplus=points.clone()
                points_yplus=points.clone()
                points_zplus=points.clone()
                points_xplus[:,0]=points_xplus[:,0]+epsilon
                points_yplus[:,1]=points_yplus[:,1]+epsilon
                points_zplus[:,2]=points_zplus[:,2]+epsilon
                points_full=torch.cat([points, points_xplus, points_yplus, points_zplus],0)

               
            sdf_full, geom_feat_full = self.forward(points_full, iter_nr)

            geom_feat=None
            if geom_feat_full is not None:            
                g_feats=geom_feat_full.chunk(4, dim=0) 
                geom_feat=g_feats[0]

            sdfs=sdf_full.chunk(4, dim=0) 
            sdf=sdfs[0]
            sdf_xplus=sdfs[1]
            sdf_yplus=sdfs[2]
            sdf_zplus=sdfs[3]

            grad_x=(sdf_xplus-sdf)/epsilon
            grad_y=(sdf_yplus-sdf)/epsilon
            grad_z=(sdf_zplus-sdf)/epsilon

            gradients=torch.cat([grad_x, grad_y, grad_z],1)


        elif method=="autograd":

            #do it with autograd
            with torch.set_grad_enabled(True):
                points.requires_grad_(True)
                sdf, geom_feat = self.forward(points, iter_nr)

                feature_vectors=None
                d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
                gradients = torch.autograd.grad(
                    outputs=sdf,
                    inputs=points,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)[0]
        else:
            print("method not known")
            exit(1)




        return sdf, gradients, geom_feat

    def get_sdf_and_curvature_1d_precomputed_gradient_normal_based(self, points, sdf_gradients,iter_nr):
        #get the curvature along a certain random direction for each point
        #does it by computing the normal at a shifted point on the tangent plant and then computing a dot produt



        #to the original positions, add also a tiny epsilon 
        nr_points_original=points.shape[0]
        epsilon=1e-4
        rand_directions=torch.randn_like(points)
        rand_directions=F.normalize(rand_directions,dim=-1)

        #instead of random direction we take the normals at these points, and calculate a random vector that is orthogonal 
        normals=F.normalize(sdf_gradients,dim=-1)
        # normals=normals.detach()
        tangent=torch.cross(normals, rand_directions)
        rand_directions=tangent #set the random moving direction to be the tangent direction now
        

        points_shifted=points.clone()+rand_directions*epsilon
        
        #get the gradient at the shifted point
        sdf_shifted, sdf_gradients_shifted, feat_shifted=self.get_sdf_and_gradient(points_shifted, iter_nr) 

        normals_shifted=F.normalize(sdf_gradients_shifted,dim=-1)

        dot=(normals*normals_shifted).sum(dim=-1, keepdim=True)
        #the dot would assign low weight importance to normals that are almost the same, and increasing error the more they deviate. So it's something like and L2 loss. But we want a L1 loss so we get the angle, and then we map it to range [0,1]
        angle=torch.acos(torch.clamp(dot, -1.0+1e-6, 1.0-1e-6)) #goes to range 0 when the angle is the same and 2pi when is opposite


        curvature=angle/(2.0*math.pi) #map to [0,1 range]

        return sdf_shifted, curvature

    def path_to_save_model(self, ckpt_folder, experiment_name, iter_nr):
        models_path=os.path.join(ckpt_folder, experiment_name, str(iter_nr), "models")
        return models_path

    def save(self, ckpt_folder, experiment_name, iter_nr):

        # models_path=os.path.join(ckpt_folder, experiment_name, str(iter_nr), "models")
        models_path=self.path_to_save_model(ckpt_folder, experiment_name, iter_nr)
        os.makedirs(models_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(models_path, "sdf_model.pt")  )

        return models_path 

class RGB(torch.nn.Module):

    def __init__(self, in_channels, boundary_primitive, geom_feat_size_in, nr_iters_for_c2f):
        super(RGB, self).__init__()

        self.in_channels=in_channels
        self.boundary_primitive=boundary_primitive
        self.geom_feat_size_in=geom_feat_size_in


        # self.pick_rand_rows= RandRowPicker()
        # self.pick_rand_pixels= RandPixelPicker(low_discrepancy=False) #do NOT use los discrepancy for now, it seems to align some of the rays to some directions so maybe it's not that good of an idea
        # self.pixel_sampler=PixelSampler()
        self.create_rays=CreateRaysModule()
        self.volume_renderer_neus = VolumeRenderingNeus()

        #create encoding
        pos_dim=in_channels
        capacity=pow(2,18) #2pow18
        nr_levels=24 
        nr_feat_per_level=2 
        coarsest_scale=1.0 
        finest_scale=0.0001 
        scale_list=np.geomspace(coarsest_scale, finest_scale, num=nr_levels)
        self.encoding=permuto_enc.PermutoEncoding(pos_dim, capacity, nr_levels, nr_feat_per_level, scale_list, appply_random_shift_per_level=True, concat_points=True, concat_points_scaling=1)           

       

        # with dirs encoded
        # self.mlp= torch.nn.Sequential(
        #     torch.nn.Linear(self.encoding.output_dims() + 25 + 3 + geom_feat_size_in, 128),
        #     torch.nn.GELU(),
        #     torch.nn.Linear(128,128),
        #     torch.nn.GELU(),
        #     torch.nn.Linear(128,64),
        #     torch.nn.GELU(),
        #     torch.nn.Linear(64,3)
        # )
        # apply_weight_init_fn(self.mlp, leaky_relu_init, negative_slope=0.0)
        # leaky_relu_init(self.mlp[-1], negative_slope=1.0)
        mlp_in_channels=self.encoding.output_dims() + 25 + 3 + geom_feat_size_in
        self.mlp=LipshitzMLP(mlp_in_channels, [128,128,64,3], last_layer_linear=True)

        self.c2f=permuto_enc.Coarse2Fine(nr_levels)
        self.nr_iters_for_c2f=nr_iters_for_c2f
        self.last_iter_nr=sys.maxsize 

        self.softplus=torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points, samples_dirs, sdf_gradients, geom_feat,  iter_nr, model_colorcal=None, img_indices=None, ray_start_end_idx=None):

     

        assert points.shape[1] == self.in_channels, "points should be N x in_channels"

        self.last_iter_nr=iter_nr

       
        window=self.c2f( map_range_val(iter_nr, 0.0, self.nr_iters_for_c2f, 0.3, 1.0   ) )

        point_features=self.encoding(points, window.view(-1))
        #dirs encoded with spherical harmonics 
        with torch.set_grad_enabled(False):
            samples_dirs_enc=PermutoSDF.spherical_harmonics(samples_dirs,5)
        #normals
        normals=F.normalize( sdf_gradients.view(-1,3), dim=1 )

       
        x=torch.cat([point_features, samples_dirs_enc, normals, geom_feat],1)
       

        x=self.mlp(x)

        if model_colorcal is not None:
            x=model_colorcal.calib_RGB_samples_packed(x, img_indices, ray_start_end_idx )
        

        x = self.sigmoid(x)
        


        return x

    def path_to_save_model(self, ckpt_folder, experiment_name, iter_nr):
        models_path=os.path.join(ckpt_folder, experiment_name, str(iter_nr), "models")
        return models_path

    def save(self, ckpt_folder, experiment_name, iter_nr):

        # models_path=os.path.join(ckpt_folder, experiment_name, str(iter_nr), "models")
        models_path=self.path_to_save_model(ckpt_folder, experiment_name, iter_nr)
        os.makedirs(models_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(models_path, "rgb_model.pt")  )

        return models_path 

    def parameters_only_encoding(self):
        params=[]
        for name, param in self.encoding.named_parameters():
            if "lattice_values" in name:
                params.append(param)
        return params

    def parameters_all_without_encoding(self):
        params=[]
        for name, param in self.named_parameters():
            if "lattice_values" in name:
                pass
            else:
                params.append(param)
        return params
                


###################NERF ################################
class NerfHash(torch.nn.Module):

    def __init__(self, in_channels, boundary_primitive, nr_iters_for_c2f):
        super(NerfHash, self).__init__()

        self.in_channels=in_channels
        self.boundary_primitive=boundary_primitive


        #create encoding
        pos_dim=in_channels
        capacity=pow(2,18) #2pow18
        nr_levels=24 
        nr_feat_per_level=2 
        coarsest_scale=1.0 
        finest_scale=0.0001 
        scale_list=np.geomspace(coarsest_scale, finest_scale, num=nr_levels)
        self.encoding=permuto_enc.PermutoEncoding(pos_dim, capacity, nr_levels, nr_feat_per_level, scale_list, appply_random_shift_per_level=True, concat_points=True, concat_points_scaling=1)       

       
        self.create_rays=CreateRaysModule()
        self.volume_renderer_nerf=VolumeRenderingNerf()

        

        self.nr_feat_for_rgb=64
        self.mlp_feat_and_density= torch.nn.Sequential(
            torch.nn.Linear(self.encoding.output_dims(), 64),
            torch.nn.GELU(),
            torch.nn.Linear(64,64),
            torch.nn.GELU(),
            torch.nn.Linear(64,64),
            torch.nn.GELU(),
            torch.nn.Linear(64,self.nr_feat_for_rgb+1) 
        )
        apply_weight_init_fn(self.mlp_feat_and_density, leaky_relu_init, negative_slope=0.0)
       
       

        self.mlp_rgb= torch.nn.Sequential(
            torch.nn.Linear(self.nr_feat_for_rgb+16, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64,3),
        )
        apply_weight_init_fn(self.mlp_rgb, leaky_relu_init, negative_slope=0.0)
        leaky_relu_init(self.mlp_rgb[-1], negative_slope=1.0)
        


        self.softplus=torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()
        self.gelu=torch.nn.GELU()



        self.c2f=permuto_enc.Coarse2Fine(nr_levels)
        self.nr_iters_for_c2f=nr_iters_for_c2f
        self.last_iter_nr=sys.maxsize

    
    # def forward(self, ray_origins, ray_dirs, ls, iter_nr, nr_samples_per_ray):
    def forward(self, samples_pos, samples_dirs, iter_nr, model_colorcal=None, img_indices=None, ray_start_end_idx=None):


        assert samples_pos.shape[1] == self.in_channels, "points should be N x in_channels"

        self.last_iter_nr=iter_nr

       
        window=self.c2f( map_range_val(iter_nr, 0.0, self.nr_iters_for_c2f, 0.3, 1.0   ) ) #helps to converge the radiance of the object in the center and not put radiance on the walls of the bounding box
        

        points=samples_pos
        point_features=self.encoding(points, window.view(-1)) 
        #dirs encoded with spherical harmonics 
        with torch.set_grad_enabled(False):
            samples_dirs_enc=PermutoSDF.spherical_harmonics(samples_dirs,4)



        #predict density without using directions
        feat_and_density=self.mlp_feat_and_density(point_features)
        density=feat_and_density[:,0:1]
        feat_rgb=feat_and_density[:,1:self.nr_feat_for_rgb+1]

        #predict rgb using directions of ray
        feat_rgb_with_dirs=torch.cat([ self.gelu(feat_rgb), samples_dirs_enc],1)
        rgb=self.mlp_rgb(feat_rgb_with_dirs)

        #activate
        density=self.softplus(density) #similar to mipnerf


        if model_colorcal is not None:
            rgb=model_colorcal.calib_RGB_samples_packed(rgb, img_indices, ray_start_end_idx )

        rgb=self.sigmoid(rgb)


        return rgb, density

    def get_only_density(self, ray_samples, iter_nr):


        # window=self.c2f(iter_nr*0.0001) #helps to converge the radiance of the object in the center and not put radiance on the walls of the bounding box
        window=self.c2f( map_range_val(iter_nr, 0.0, self.nr_iters_for_c2f, 0.3, 1.0   ) ) #helps to converge the radiance of the object in the center and not 



        #given the rays, create points
        points=ray_samples.view(-1,ray_samples.shape[-1])

      


        point_features=self.encoding(points, window.view(-1))  


        #predict density without using directions
        feat_and_density=self.mlp_feat_and_density(point_features)
        density=feat_and_density[:,0:1]
        density=self.softplus(density) #similar to mipnerf

        return density

    def path_to_save_model(self, ckpt_folder, experiment_name, iter_nr):
        models_path=os.path.join(ckpt_folder, experiment_name, str(iter_nr), "models")
        return models_path

    def save(self, ckpt_folder, experiment_name, iter_nr, additional_name=None):

        # models_path=os.path.join(ckpt_folder, experiment_name, str(iter_nr), "models")
        models_path=self.path_to_save_model(ckpt_folder, experiment_name, iter_nr)
        os.makedirs(models_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(models_path, "nerf_hash_model"+str(additional_name or '')+".pt")  )

        return models_path


class DeferredRender(torch.nn.Module):

    def __init__(self, tex_size, tex_nr_channels, nr_textures, use_mlp, use_unet):
        super(DeferredRender, self).__init__()

        self.use_mlp=use_mlp
        self.use_unet=use_unet

        
        #make texture
        # self.texture= torch.randn( 1, tex_nr_channels, tex_size, tex_size  )*1e-4
        # self.texture=torch.nn.Parameter(self.texture)

        #make various textures
        self.nr_textures=nr_textures
        self.max_tex_size=tex_size
        self.textures = nn.ParameterList()
        for i in range(self.nr_textures):
            cur_tex_size=tex_size//pow(2,i)
            print("cur_tex_size", cur_tex_size)
            # self.textures.append( torch.nn.Parameter( torch.randn( 1, tex_nr_channels, cur_tex_size, cur_tex_size  )*1e-4  )   )
            self.textures.append( torch.nn.Parameter( torch.randn( 1, tex_nr_channels, cur_tex_size, cur_tex_size  )*0  )   )




        if self.use_mlp:
            self.mlp= nn.Sequential(
                LinearWN(tex_nr_channels, 32),
                torch.nn.Mish(),
                LinearWN(32,32),
                torch.nn.Mish(),
                LinearWN(32,32),
                torch.nn.Mish(),
                LinearWN(32,3)
            )
            apply_weight_init_fn(self, leaky_relu_init, negative_slope=0.0)
            leaky_relu_init(self.mlp[-1], negative_slope=1.0)

        if self.use_unet:
            self.unet=UNet(in_channels=tex_nr_channels, start_channels=8, nr_downsamples=2, compression_rate=1.0, out_channels=3, max_nr_channels=128)
        




        self.rgb_padding = 0.001  # Padding added to the RGB outputs.


        self.sigmoid = torch.nn.Sigmoid()



        self.c2f=Coarse2Fine(self.nr_textures)
        # self.c2f_tex=Coarse2FineTexture(iterations_to_finish=1000)
        self.c2f_tex=Coarse2FineImg(iterations_to_finish=5000)

    
    def forward(self, uv_tensor, iter_nr):

        # window=self.c2f(iter_nr*0.0003)
        # print("window", window)

        uv_tensor_channels_last=uv_tensor.permute(0,2,3,1)
        uv_tensor_channels_last_11=uv_tensor_channels_last*2-1.0
        # print("uv_tensor_channels_last", uv_tensor_channels_last.shape)

        # tex=self.texture.permute(0,2,3,1).squeeze(0)
        # if self.training:
        #     tex=self.c2f_tex(tex)
        # tex=tex.permute(2,0,1).unsqueeze(0)


        # if self.training: #backward hood cannot be applied when we are not training
            # tex=self.c2f_tex(self.texture)
        # else:
            # tex=self.texture
        # tex=self.texture

        #slice one texture
        # sampled_feat=torch.nn.functional.grid_sample(tex, uv_tensor_channels_last_11, mode='bilinear')
        # x=sampled_feat


        #slice multiple textures
        x=0
        for i in range(self.nr_textures):
            # x+= window[i]* torch.nn.functional.grid_sample(self.textures[i], uv_tensor_channels_last_11, mode='bilinear')
            x+=  torch.nn.functional.grid_sample(self.textures[i], uv_tensor_channels_last_11, mode='bilinear')

        x=x.float()

        # print("x is ", x.type())


        if self.use_unet:
            x=self.unet(x)

        # x = self.sigmoid(x)
      


        return x

    def save(self, root_folder, experiment_name, iter_nr):

        models_path=os.path.join(root_folder,"checkpoints/", experiment_name, str(iter_nr), "models")
        os.makedirs(models_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(models_path, "deferred_render_model.pt")  )



class Colorcal(torch.nn.Module):
    def __init__(self, nr_cams, idx_with_fixed_calib):
        super(Colorcal, self).__init__()


        #we need an image to have no calibration (so the weight=1 and bias=0 for it)
        #if we don't have this, the network is would be allowed to change the weight and bias arbitraraly and the RGB network would compensate and therefore would probably generalize poorly to novel views
        self.idx_with_fixed_calib=idx_with_fixed_calib 

        #instead of storing the weight which should be 1, we store the weight delta from 1. This allows us to use weight decay for this module which will keep the weight_delta close to zero 
        self.weight_delta = torch.nn.Parameter(
                torch.zeros(nr_cams, 3))
        self.bias = torch.nn.Parameter(
                torch.zeros(nr_cams, 3))

    def calib_RGB_samples_packed(self, rgb_samples, per_pixel_img_indices, ray_start_end_idx):
        # rgb_samples_contains the rgb for eahc sample in a nr_samples_total x3. This is due tot he fact that we sue an occupancy grid so eveyr ray has different number of samples
        #RGB_linear is nr_rays, nr_samples,3
        assert rgb_samples.dim()==2, "RGB_samples should have 2 dimensions corresponding to nr_samples_total, 3"
        assert rgb_samples.shape[1]==3, "RGB_samples should be nr_samples_total,3"
        #each pixel may be sampled from a different image
        assert per_pixel_img_indices.dim()==1, "per_pixel_img_indices should have 1 dimensions"


        #gather 3 weights and 3 biases for each pixel
        weights_delta_per_pixel=torch.index_select(self.weight_delta, 0, per_pixel_img_indices.long()) #nr_rays x3
        weights_per_pixel=1.0+weights_delta_per_pixel
        bias_per_pixel=torch.index_select(self.bias, 0, per_pixel_img_indices.long()) #nr_rays x3

        #for the camera that is fixed, it's weights should be 1 and bias should be 0
        fixed_pixels=per_pixel_img_indices==self.idx_with_fixed_calib
        weights_per_pixel[fixed_pixels,:]=1.0
        bias_per_pixel[fixed_pixels,:]=0.0


        #get the nr of samples per_ray
        nr_samples_per_ray=ray_start_end_idx[:,1:2]-ray_start_end_idx[:,0:1] 
        #repeat each weight and each bias, as many samples as we have for each ray
        weights_per_pixel=torch.repeat_interleave(weights_per_pixel, nr_samples_per_ray.view(-1), dim=0)
        bias_per_pixel=torch.repeat_interleave(bias_per_pixel, nr_samples_per_ray.view(-1), dim=0)

        rgb_samples=rgb_samples*weights_per_pixel+bias_per_pixel

        return rgb_samples

    def path_to_save_model(self, ckpt_folder, experiment_name, iter_nr):
        models_path=os.path.join(ckpt_folder, experiment_name, str(iter_nr), "models")
        return models_path

    def save(self, ckpt_folder, experiment_name, iter_nr):

        # models_path=os.path.join(root_folder,"checkpoints/", experiment_name, str(iter_nr), "models")
        models_path=self.path_to_save_model(ckpt_folder, experiment_name, iter_nr)
        os.makedirs(models_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(models_path, "colorcal_model.pt")  )

   