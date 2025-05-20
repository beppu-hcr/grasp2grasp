import open3d as o3d
import numpy as np
from scipy.ndimage import distance_transform_edt

##############################################################################
#                                KD-TREE                                     #
##############################################################################

def build_kdtree_for_object(object_points):
    """
    Build and return an Open3D KD-tree from the object's point cloud.
    
    :param object_points: (N,3) numpy array of object point positions
    :return:             (kd_tree, object_pcd)
    """
    object_pcd = o3d.geometry.PointCloud()
    object_pcd.points = o3d.utility.Vector3dVector(object_points)
    
    kd_tree = o3d.geometry.KDTreeFlann(object_pcd)
    return kd_tree, object_pcd

def find_object_contact_points_kdtree(kd_tree, object_pcd, hand_points, threshold):
    """
    Find all object points that lie within 'threshold' distance from
    ANY point in 'hand_points'.
    
    Returns a (K, 3) array of object contact points (subset of object_points).
    """
    # Convert to numpy array for indexing
    object_points = np.asarray(object_pcd.points)
    
    contacted_indices = set()
    for hp in hand_points:
        # Radius search: returns (count, list_of_indices, list_of_sq_distances)
        _, idxs, _ = kd_tree.search_radius_vector_3d(hp, threshold)
        for i in idxs:
            contacted_indices.add(i)
    
    # Extract the subset of object_points
    contacted_indices = list(contacted_indices)
    contact_points = object_points[contacted_indices]
    return contact_points

def sample_fixed_points(points, num_points):
    """
    Sample the input point cloud so that it has exactly num_points.
    If points has more than num_points, sample without replacement.
    If fewer, include all original points and sample additional points (with replacement)
    to reach num_points, then shuffle the result.
    """
    N = points.shape[0]
    if N >= num_points:
        indices = np.random.choice(N, num_points, replace=False)
        return points[indices]
    else:
        # Include all original points
        additional = points[np.random.choice(N, num_points - N, replace=True)]
        combined = np.concatenate([points, additional], axis=0)
        # Shuffle so the duplicated points are not always at the end
        np.random.shuffle(combined)
        return combined


##############################################################################
#                         EXAMPLE USAGE / TIMING                             #
##############################################################################

if __name__ == "__main__":
    import time
    # --------------------------------------------------------------------
    # 1) READ OBJECT POINT CLOUD (XYZ)
    #    Example "object.xyz" contains Nx3 numeric values
    # --------------------------------------------------------------------
    object_base_path = f'/diskstation/XXX/data/multigripper_grasp_data/Dataset/Object_Models/YCB'
    object_path = f'{object_base_path}/003_cracker_box/points.xyz'
    object_points = np.loadtxt(object_path, delimiter=' ')
    # object_points = np.loadtxt("object.xyz")  # shape: (N,3)

    # Suppose we have multiple hand point clouds for the same object in NPY format
    # e.g. ["hand1.npy", "hand2.npy", ...]
    hand_files = ["test_003_cracker_box_2.npy", "test_003_cracker_box_5.npy", "test_003_cracker_box_8.npy"]  # etc.

    # Set a distance threshold
    threshold = 0.01

    # --------------------------------------------------------------------
    # 2) BUILD KD-TREE (ONE-TIME)
    # --------------------------------------------------------------------
    start = time.time()
    kd_tree, obj_pcd = build_kdtree_for_object(object_points) # shape: (N,3)
    end = time.time()
    print(f"[KD-Tree] Built in {end - start:.4f}s")

    # --------------------------------------------------------------------
    # 4) PROCESS MULTIPLE HAND POINT CLOUDS
    # --------------------------------------------------------------------
    for hand_file in hand_files:
        # Load hand points from .npy file
        hand_points = np.load(f"test_data/{hand_file}")  # shape: (M,3)
        
        # --- KD-Tree Query ---
        start = time.time()
        contact_points_dt = find_object_contact_points_kdtree(kd_tree,
                                                           obj_pcd,
                                                           hand_points,
                                                           threshold)
        end = time.time()
        print(f"[{hand_file}] KD-tree => #contacts={len(contact_points_dt)} in {end - start:.4f}s")
        print(contact_points_dt.shape)
        contact_points_dt = sample_fixed_points(contact_points_dt, 256)
        print(contact_points_dt.shape)

        # np.save(f"test_data/{hand_file}_contacts.npy", contact_points_dt)
        # save xyz to visualize in meshlab
        # np.savetxt(f"test_data/{hand_file}_contacts.xyz", contact_points_dt, delimiter=' ')

