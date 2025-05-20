import os
import pandas as pd
import numpy as np
import torch
from pytorch3d.transforms import matrix_to_rotation_6d, quaternion_to_matrix

def process_metadata_file(obj_path):
    """Helper function to load metadata from a single object directory."""
    meta_file = os.path.join(obj_path, 'metadata.parquet')
    if not os.path.exists(meta_file):
        return None, None
    df = pd.read_parquet(meta_file)
    # Convert entire DataFrame to a list of dicts in one go
    samples = df.to_dict(orient='records')
    # Add additional fields to each sample
    for idx, sample in enumerate(samples):
        sample['base_dir'] = obj_path
        sample['grasp_idx'] = idx
    return obj_path, samples

def extract_posetheta_from_metadata(obj_path):
    """
    Extract pose and theta from metadata files in the given object path.
    Returns a list of tuples (pose, theta) for each sample in the metadata.
    """
    meta_file = os.path.join(obj_path, 'metadata.parquet')
    if not os.path.exists(meta_file):
        return None
    df = pd.read_parquet(meta_file)
    # Convert entire DataFrame to a list of tuples in one go
    poses = np.stack(df['hand_pose'].tolist())
    # poses_tensor = torch.tensor(poses)
    thetas = np.stack(df['graspit_dofs'].tolist())
    # thetas_tensor = torch.tensor(thetas)
    return poses, thetas

def get_hand_param(pose, theta):
    batch_size = pose.shape[0]
    if pose.shape[1] == 9:
        return torch.cat([pose, theta], dim=1)
    elif pose.shape[1] == 7:
        rot_matrix = quaternion_to_matrix(pose[:, 3:7])
        rot_6d = matrix_to_rotation_6d(rot_matrix)
        return torch.cat([pose[:, :3], rot_6d, theta], dim=1)
    else:
        raise ValueError("Pose should be of shape (batch_size x 7) or (batch_size x 9)")

def sample_fixed_points(points, num_points):
    """
    Sample the input point cloud so that it has exactly num_points.
    If points has more than num_points, sample without replacement.
    If fewer, include all original points and sample additional points (with replacement)
    to reach num_points, then shuffle the result.
    """
    N = points.shape[0]
    if N == num_points:
        return points
    elif N > num_points:
        indices = np.random.choice(N, num_points, replace=False)
        return points[indices]
    else:
        # Include all original points
        additional = points[np.random.choice(N, num_points - N, replace=True)]
        combined = np.concatenate([points, additional], axis=0)
        # Shuffle so the duplicated points are not always at the end
        np.random.shuffle(combined)
        return combined
    
