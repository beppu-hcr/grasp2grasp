from easydict import EasyDict as edict

__C = edict()

model_cfg = __C

# generation
__C.unet = edict()
__C.unet.pc_global_dim = 128  # lion: latent_pts.style_dimpc_checkpoint
__C.unet.hand_param_dim = 768
__C.unet.pc_local_pts = 2048
__C.unet.pc_local_pts_dim = 4

# contact
__C.unet.gen_contact = True

# UViT
__C.unet.embed_dim = 512
__C.unet.pos_drop_rate = 0.
__C.unet.num_heads = 8
__C.unet.mlp_ratio = 4
__C.unet.qkv_bias = False
__C.unet.qk_scale = None
__C.unet.drop_rate = 0.
__C.unet.attn_drop_rate = 0.1
__C.unet.use_checkpoint = True
__C.unet.depth = 12
__C.unet.mlp_time_embed = False
__C.unet.temb_scale = 1000.0

# FM
__C.fm = edict()
__C.fm.ot_method = 'exact'
__C.fm.sigma = 0.1
__C.fm.ot_cost = 'chamfer'

def get_model_cfg():
    return model_cfg
