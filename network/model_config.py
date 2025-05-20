from easydict import EasyDict as edict

__C = edict()

model_cfg = __C

__C.UGG = edict()

# generation
__C.UGG.unet = edict()
__C.UGG.unet.pc_global_dim = 128  # lion: latent_pts.style_dimpc_checkpoint
__C.UGG.unet.hand_param_dim = 768
__C.UGG.unet.pc_local_pts = 2048
__C.UGG.unet.pc_local_pts_dim = 4

# contact
__C.UGG.unet.gen_contact = True

# UViT
__C.UGG.unet.embed_dim = 512
__C.UGG.unet.pos_drop_rate = 0.
__C.UGG.unet.num_heads = 8
__C.UGG.unet.mlp_ratio = 4
__C.UGG.unet.qkv_bias = False
__C.UGG.unet.qk_scale = None
__C.UGG.unet.drop_rate = 0.
__C.UGG.unet.attn_drop_rate = 0.1
__C.UGG.unet.use_checkpoint = True
__C.UGG.unet.depth = 12
__C.UGG.unet.mlp_time_embed = False
__C.UGG.unet.temb_scale = 1000.0

# FM
__C.UGG.fm = edict()
__C.UGG.fm.ot_method = 'exact'
__C.UGG.fm.sigma = 0.1
__C.UGG.fm.ot_cost = 'chamfer'

def get_model_cfg():
    return model_cfg.UGG
