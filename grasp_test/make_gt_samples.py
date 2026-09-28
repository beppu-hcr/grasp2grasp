"""Build a samples.pkl from ground-truth dataset grasps, for smoke-testing isaac_test_right.py
without running sample.py. Run from the repo root:
    python grasp_test/make_gt_samples.py --hand Allegro --out_dir logs/isaac_gt_test
"""
import os
import sys
import json
import pickle
import argparse

import numpy as np
import torch

sys.path.append(os.getcwd())
from dataset.data_utils import extract_posetheta_from_metadata, get_hand_param


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/grasp_data')
    parser.add_argument('--hand', default='Allegro')
    parser.add_argument('--split', default='test')
    parser.add_argument('--num_objects', type=int, default=3)
    parser.add_argument('--num_samples', type=int, default=16)
    parser.add_argument('--out_dir', default='./logs/isaac_gt_test')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    with open(os.path.join(args.data_dir, args.hand, 'split.json')) as f:
        obj_dirs = json.load(f)[args.split][:args.num_objects]

    sample_qpos = {}
    for obj_dir in obj_dirs:
        obj_id = os.path.basename(obj_dir)
        poses, thetas = extract_posetheta_from_metadata(os.path.join(args.data_dir, args.hand, obj_id))
        idx = rng.choice(len(poses), min(args.num_samples, len(poses)), replace=False)
        param = get_hand_param(torch.tensor(poses[idx]), torch.tensor(thetas[idx]))
        sample_qpos[obj_id] = param.numpy()
        print(f'{obj_id}: {sample_qpos[obj_id].shape}')

    os.makedirs(args.out_dir, exist_ok=True)
    out = {
        'method': f'gt_{args.hand}',
        'sample_qpos': sample_qpos,
        'sample_iou': {'mean': float('nan')},
        'sample_std': float(np.mean([q[:, 9:].std(axis=0).mean() for q in sample_qpos.values()])),
    }
    with open(os.path.join(args.out_dir, 'samples.pkl'), 'wb') as f:
        pickle.dump(out, f)
    print(f'Saved to {os.path.join(args.out_dir, "samples.pkl")}')


if __name__ == '__main__':
    main()
