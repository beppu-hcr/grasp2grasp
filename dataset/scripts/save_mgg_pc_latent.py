import sys
import os
from os.path import join as pjoin
base_dir = os.path.dirname(__file__)
sys.path.append(pjoin(base_dir, '..'))
sys.path.append(pjoin(base_dir, '../LION/'))

import pickle
import yaml
from easydict import EasyDict as edict
import torch
import trimesh
import numpy as np
import glob

mesh_path = "/data/mgg_pc/objects/obj"
save_path = "/data/mgg_pc/objects/object_pc"
pc_args = "../checkpoints/lion/aeb159h_hvae_lion_B32/cfg.yml"

# scale_list = [1.0] #[0.1] # [0.06, 0.08, 0.1, 0.12, 0.15]
scale = 6.6

cfg = edict()
cfg.MODEL = edict()
cfg.MODEL.diffusion = edict()
cfg.MODEL.diffusion.pc_global_dim = 128
cfg.DATA = edict()
cfg.DATA.PC_NUM_POINTS = 2048

device = torch.device('cuda:0')

import LION.vae_adain as vae_adain

with open(pc_args, 'r') as f:
    pc_args = edict(yaml.full_load(f))
pc_latent_model = vae_adain.Model(cfg, pc_args)

pc_checkpoint = "../checkpoints/lion/aeb159h_hvae_lion_B32/checkpoints/epoch_5999_iters_1667999.pt"

print('Load vae_checkpoint: {}', pc_checkpoint)
vae_ckpt = torch.load(pc_checkpoint)
vae_weight = vae_ckpt['model']
pc_latent_model.load_state_dict(vae_weight)

pc_latent_model = pc_latent_model.to(device=device)
pc_latent_model.eval()

mesh_list = glob.glob(os.path.join(mesh_path, "*.obj"))

for mesh_file in mesh_list:
    print(f'Processing {mesh_file}')
    mesh = trimesh.load(mesh_file, force='mesh')
    mesh_id = os.path.basename(mesh_file).split('.')[0]
    # mesh_grasp_dir = os.path.join(mesh_path, f"shadowhand_{i}.obj")
    mesh_save_path = os.path.join(save_path, f"{mesh_id}")
    if not os.path.exists(mesh_save_path):
        os.makedirs(mesh_save_path)

    # mesh = trimesh.load(mesh_grasp_dir, force='mesh')
    points_ori_list = []
    for j in range(16):
        samples, fid = mesh.sample(2048, return_index=True)
        points_ori_list.append(samples)
        # np.savetxt(os.path.join(mesh_path, f"pc/pc_2048_{mesh_file}_{j:03d}.xyz"), samples, delimiter=' ')
        # np.savetxt(f"./obj_test/pc_2048_{mesh_id}_{j:03d}.xyz", samples * 6.6, delimiter=' ')
        np.save(os.path.join(mesh_save_path, f"pc_2048_{j:03d}.npy"), samples)
    points_ori = np.stack(points_ori_list, axis=0)  # [16, N, 3]

    points_tensor = torch.tensor(points_ori, dtype=torch.float32)
    
    points_tensor = points_tensor.to(device)  # [16, N, 3]
    
    points_tensor *= scale
    
    # print(points_tensor.shape)
    # output = pc_latent_model.recont(points_tensor)
    _, _, latent_list = pc_latent_model.encode(points_tensor)

    latent_list_np = [[t.detach().cpu().numpy().reshape(16, -1) for t in li] for li in latent_list]

    for j in range(16):
        v_new = [[tt[j] for tt in t] for t in latent_list_np]
        # for t in v_new:
        #     for tt in t:
        #         print(tt.shape)
        with open(os.path.join(mesh_save_path, f"pc_norm_latent_LION_{j:03d}.pk"), 'wb') as f:
            pickle.dump(v_new, f, protocol=pickle.HIGHEST_PROTOCOL)

    # x_pred = output['x_0_pred']
    
    # for j in range(3):
    #     # np.save(os.path.join(mesh_path, f"pc/pc_2048_{i}_{j:03d}_recon.npy"), x_pred[j].cpu().detach().numpy())
    #     np.savetxt(f"./obj_test/pc_2048_{mesh_id}_{j:03d}_recon.xyz", x_pred[j].cpu().detach().numpy(), delimiter=' ')
    # latent_list = output['latent_list']
    # print(latent_list[0][0].shape, latent_list[1][0].shape)

    # _, _, latent_list = pc_latent_model.encode(points_tensor)
    # print(latent_list[0][0].shape, latent_list[0][1].shape, latent_list[0][2].shape)
    # print(latent_list[1][0].shape, latent_list[1][1].shape, latent_list[1][2].shape)
    
