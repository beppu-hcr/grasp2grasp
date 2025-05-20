import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
torch.set_num_threads(20)
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from dataset.mgg_dataset import GraspDataset
from network.autoencoder.autoencoder import Autoencoder
import argparse
import json

def save_latent_features_by_object(args, root_dir, hand, split='train', batch_size=512, device='cuda'):
    # Create dataset instance (it does not load everything into memory at once)
    dataset = GraspDataset(root_dir=root_dir, hand=hand, split=split, val_subsample_ratio=1.0)
    model = Autoencoder(args = args,
        obj_inchannel=args.obj_inchannel,
        cvae_encoder_sizes=args.encoder_layer_sizes,
        cvae_decoder_sizes=args.decoder_layer_sizes).to(device).eval()
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   f"logs/autoencoder/mgg/{args.file_name}/model_best_test.pth")
    model.load_state_dict(torch.load(checkpoint_path)['network'])
    model = model.to(device)
    model.eval()

    # Group indices by object directory.
    object_to_indices = {}
    for idx, sample_meta in enumerate(dataset.samples):
        obj_dir = sample_meta['base_dir']
        if obj_dir not in object_to_indices:
            object_to_indices[obj_dir] = []
        object_to_indices[obj_dir].append(idx)

    print(f"Processing {len(object_to_indices)} objects...")

    # Process each object one by one.
    for i, (obj_dir, indices) in enumerate(object_to_indices.items()):
        print(f"Processing object {i+1}/{len(object_to_indices)}: {os.path.basename(obj_dir)} ({len(indices)} grasps)")
        # Skip if features already exist.
        feature_file = os.path.join(obj_dir, f'hand_pc_features.pt')
        if os.path.exists(feature_file):
            print(f"Features already exist at {feature_file}, skipping...")
            continue

        # Create a Subset for all samples in this object.
        obj_subset = Subset(dataset, indices)
        obj_loader = DataLoader(
            obj_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True
        )

        latents = []
        grasp_idx = []
        with torch.no_grad():
            for batch in obj_loader:
                # Process hand point cloud with your VAE model.
                hand_pc = batch['hand_pc'].float().to(device)
                hand_feature, _, _ = model.hand_encoder(hand_pc.permute(0,2,1))
                # pointnet
                latent = model.encoder(hand_feature)
                latents.append(latent.cpu())
                grasp_idx.extend(batch['grasp_idx'].tolist())
                # print(batch['grasp_idx'].tolist())

        # Concatenate latent vectors for all grasps in the object.
        latents = torch.cat(latents, dim=0)  # (num_grasps, latent_dim)
        # print(grasp_idx)
        assert grasp_idx == list(range(len(grasp_idx)))

        # Save the latent features in the object's directory.
        save_path = os.path.join(obj_dir, f'hand_pc_features.pt')
        torch.save(latents, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, help='Path to the JSON config file')
    parser.add_argument('--split', type=str, help='split')
    args = parser.parse_args()

    with open(args.config, 'r') as configfile:
        config = json.load(configfile)
    for key, value in config.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)

    args = parser.parse_args()

    if args.split == 'all':
        splits = ['train', 'val', 'test', 'reserved']
        for split in splits:
            save_latent_features_by_object(
                args,
                root_dir='/data/XXX/data/grasp_data',
                hand=args.hand,
                split=split,
                batch_size=496,
                device='cuda'
            )
    else:
        if 'train' in args.split:
            save_latent_features_by_object(
                args,
                root_dir='/data/XXX/data/grasp_data',
                hand=args.hand,
                split='train',
                batch_size=496,
                device='cuda'
            )
        if 'val' in args.split:
            save_latent_features_by_object(
                args,
                root_dir='/data/XXX/data/grasp_data',
                hand=args.hand,
                split='val',
                batch_size=496,
                device='cuda'
            )
        if 'test' in args.split:
            save_latent_features_by_object(
                args,
                root_dir='/data/XXX/data/grasp_data',
                hand=args.hand,
                split='test',
                batch_size=496,
                device='cuda'
            )
        if 'reserved' in args.split:
            save_latent_features_by_object(
                args,
                root_dir='/data/XXX/data/grasp_data',
                hand=args.hand,
                split='reserved',
                batch_size=496,
                device='cuda'
            )
