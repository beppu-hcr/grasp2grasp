import os
import json
import argparse
import torch
import torchsde
import numpy as np
import shutil
import pickle
from dataset.mgg_test_dataset import TestGraspDataset
from network.autoencoder.autoencoder import Autoencoder
from network.sbfm_trainer import LitGraspModel
from network.fm.sde import SDE
from network.model_config import get_model_cfg
from pytorch3d.ops import sample_farthest_points
from grippers.HandLayer import grabHandLayer
from utils.loss import set_random_seed
import trimesh
from utils.utils_gwh import get_hull, monte_carlo_iou_1to1_6d, simplify_mesh

def main():
    parser = argparse.ArgumentParser(description="Visualize hand mesh from saved model")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the JSON config file')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Dataset split to visualize (default: test)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device id to use (default: "0")')
    parser.add_argument('--num_objects', type=int, default=0,
                        help='number of objects to visualize (default: 3)')
    parser.add_argument('--num_samples', type=int, default=32,
                        help='number of samples per object (default: 4)')
    args = parser.parse_args()

    # Load additional config parameters from JSON and update args
    with open(args.config, 'r') as f:
        config = json.load(f)
    for key, value in config.items():
        setattr(args, key, value)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        print(f"Using GPU device {args.gpu}")

    # Set random seed for reproducibility
    set_random_seed(args.seed)
    N = args.num_samples
    S = args.num_objects

    # Initialize the model and move to device
    model = Autoencoder(
        args=args,
        obj_inchannel=args.obj_inchannel,
        cvae_encoder_sizes=args.encoder_layer_sizes,
        cvae_decoder_sizes=args.decoder_layer_sizes
    ).to(device)

    # Load the saved checkpoint
    checkpoint = torch.load(args.vae_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['network'])
    model.eval()

    cfg = get_model_cfg()
    trained_model = LitGraspModel.load_from_checkpoint(args.fm_checkpoint, map_location=device, cfg=cfg)
    ema_model = trained_model.ema_model
    ema_score_model = trained_model.ema_score_model
    ema_model.eval()
    ema_score_model.eval()

    sde = SDE(ema_model, ema_score_model, input_size=(128 + 2048*4 + 768 + 5*3,), sigma=args.sigma).to(device)
    sde.eval()

    # Initialize the hand layer model (for mesh reconstruction)
    hand1_layer = grabHandLayer(hand_name=args.hand1, device=device)
    hand2_layer = grabHandLayer(hand_name=args.hand2, device=device)

    # Load the dataset (using the specified split)
    dataset = TestGraspDataset(root_dir=args.data_dir, hand1=args.hand1, hand2=args.hand2,
                                split=args.split, contact=True, batch_size=N)
    if S == 0:
        S = len(dataset)
        sample_idxs = np.arange(S)
        print(f"Using all {S} samples from the dataset.")
    else:
        sample_idxs = np.random.randint(len(dataset), size=S)
    # if args.index >= len(dataset):
    #     raise IndexError(f"Index {args.index} is out of range for dataset of size {len(dataset)}")

    x0 = torch.zeros((N*S, 128 + 2048*4 + 768 + 5*3)).to(device)
    hand1_param = torch.zeros((N*S, dataset[0]["hand1_param"].shape[1])).to(device)
    objects = []
    for i, sample_idx in enumerate(sample_idxs):
        sample = dataset[sample_idx]

        # Prepare inputs (add batch dimension)
        object_global_latent = sample["object_global_latent"].to(device) # [B, D1]
        B = object_global_latent.shape[0]
        object_local_latent = sample["object_local_latent"].to(device) # [B, ND2]
        hand1_latent = sample["hand1_latent"].to(device) # [B, Dh]
        hand1_contact = sample["contact_pc_1"].to(device) # [B, N, 3]
        hand1_contact = sample_farthest_points(hand1_contact, K=5)[0] # [B, 5, 3]
        hand1_contact = hand1_contact.reshape(B, -1) # [B, 5*3]

        # [B, D1 + ND2 + Dh + 5*3]
        x0[i*N:(i+1)*N, :] = torch.cat((object_global_latent, object_local_latent, hand1_latent, hand1_contact), dim=1)

        hand1_param[i*N:(i+1)*N, :] = sample["hand1_param"].to(device) # [B, 9 + Dh]
        objects.append(sample["obj_id"])

    # Generate ground truth hand mesh and reconstruction using the network
    with torch.no_grad():
        # Ground truth hand mesh from the hand layer
        gt_hand_xyz = hand1_layer(hand1_param[:, :9], hand1_param[:, 9:])
        # Forward pass through the SDE:
        # 1. SDE forward
        sde_traj = torchsde.sdeint(
            sde,
            x0,
            ts=torch.linspace(0, 1, 10),
            dt = 0.005,
            method="euler",
        )
        # 2. Decode to get hand parameters
        predicted_z = sde_traj[-1][:, 128+2048*4:128+2048*4+768] # [B, Dh]
        hand2_param = model.decoder(predicted_z) # [B, 9 + num_joints]
        # 3. Convert parameters to mesh (assumes first 9 values and remaining values represent different aspects)
        recon_xyz = hand2_layer(hand2_param[:, :9], hand2_param[:, 9:])
        hand1_face = hand1_layer.concatenated_faces
        hand2_face = hand2_layer.concatenated_faces

        hand1_verts_np = gt_hand_xyz.cpu().numpy()
        hand2_verts_np = recon_xyz.cpu().numpy()

    # print("Hand 1 mesh shape:", gt_hand_xyz.shape)
    # print("Hand 2 mesh shape:", recon_xyz.shape)

    out_dict = {
        'method': args.file_name,
        'sample_qpos': {},
        'sample_iou': {},
        'sample_hull': {},
        'sample_std': 0,
    }
    out_dir = f"./logs/diffusion_ddp/mgg/{args.file_name}/test_output"
    os.makedirs(out_dir, exist_ok=True)

    out_dict['sample_std'] = np.mean(np.std(hand2_param.cpu().numpy()[:, 9:], axis=0))
    # print(f"Mean std of hand2 param: {np.mean(np.std(hand2_param.cpu().numpy()[:, 9:], axis=0))}")
    for i in range(S):
        object_id = objects[i]
        # save_dir = f"./test_output/fm_pose/{object_id}"
        save_dir = os.path.join(out_dir, "vis", object_id)
        os.makedirs(save_dir, exist_ok=True)
        shutil.copy(f"/data/XXX/data/mgg_pc/objects/obj/{object_id}.obj", os.path.join(save_dir, f"{object_id}.obj"))

        hand_object_param = hand2_param[i*N:(i+1)*N, :].cpu().numpy() # [B, 9 + num_joints]
        # out_dict['sample_qpos'][object_id] = hand_object_param
        # out_dict['sample_std'][object_id] = np.mean(np.std(hand_object_param[:, 9:], axis=0))
        

        obj_mesh = trimesh.load(f"/data/XXX/data/mgg_pc/objects/obj/{object_id}.obj")
        obj_mesh = simplify_mesh(obj_mesh, target_faces=5000)

        hulls_1 = []
        hulls_2 = []
        hand_pq = []
        for j in range(N):
            sample_idx = i * N + j
            # Visualize the meshes.
            # The vis_hand function is assumed to take a numpy array of shape [B, N, 3] and an optional title.
            recon_hand_mesh = trimesh.Trimesh(vertices=hand2_verts_np[sample_idx], faces=hand2_face.cpu().numpy())
            gt_hand_mesh = trimesh.Trimesh(vertices=hand1_verts_np[sample_idx], faces=hand1_face.cpu().numpy())
            recon_hand_mesh.export(os.path.join(save_dir, f"{args.hand2}_{j}.obj"))
            gt_hand_mesh.export(os.path.join(save_dir, f"{args.hand1}_{j}.obj"))

            hull_1 = get_hull(hand1_verts_np[sample_idx], obj_mesh, padding=0.008, mu=2.0)
            hull_2 = get_hull(hand2_verts_np[sample_idx], obj_mesh, padding=0.008, mu=2.0)
            if hull_1 is not None and hull_2 is not None:
                hulls_1.append(hull_1)
                hulls_2.append(hull_2)
                hand_pq.append(hand_object_param[[j], :])
            out_dict['sample_hull'][object_id] = (hulls_1, hulls_2)
            # obj_pc_np = obj_pc.cpu().numpy()
            # np.savetxt(f"test_output/kl/human/obj_pc_{sample_idx}.xyz", obj_pc_np[0], delimiter=" ")
        out_dict['sample_qpos'][object_id] = np.concatenate(hand_pq, axis=0)
        # with torch.no_grad():
        #     out_dict['sample_iou'][object_id] = monte_carlo_iou_1to1_6d(hulls_1, hulls_2, samples=500_000, device=device).cpu().numpy()
    pickle.dump(out_dict, open(os.path.join(out_dir, "samples.pkl"), 'wb'))

if __name__ == '__main__':
    main()
