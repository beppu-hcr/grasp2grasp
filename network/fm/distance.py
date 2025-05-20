import torch
from chamfer_distance import ChamferDistance as chamfer_dist
from pytorch3d.loss import chamfer_distance
import numpy as np
from scipy.spatial import ConvexHull

chd = chamfer_dist()

def subsample_point_clouds(batch_points, K):
    """
    Subsamples each point cloud in a batch from (B, N, 3) to (B, K, 3) using a
    vectorized approach without an explicit loop.
    
    Args:
        batch_points: Tensor of shape (B, N, 3)
        K: int, number of points to subsample
    Returns:
        A tensor of shape (B, K, 3) where each point cloud has been randomly subsampled.
    """
    B, N, _ = batch_points.shape
    # Generate random numbers for each point in each cloud
    rand = torch.rand(B, N, device=batch_points.device)
    # Get indices that would sort each point cloud's random values, then take the first K indices.
    indices = torch.argsort(rand, dim=1)[:, :K]  # shape: (B, K)
    # Expand indices to have the last dimension for the 3 coordinates.
    indices = indices.unsqueeze(2).expand(B, K, 3)
    # Gather along the points dimension
    subsampled = torch.gather(batch_points, 1, indices)
    return subsampled

def compute_cross_chamfer_distance(batch1, batch2, K=None):
    """
    Computes the cross Chamfer distance between each pair of point clouds from two batches.
    If K is provided, each point cloud is first subsampled to K points.
    
    Args:
        batch1: Tensor of shape (B, N, 3)
        batch2: Tensor of shape (B, N, 3)
        K: Optional int specifying the number of points to subsample to.
    
    Returns:
        A tensor of shape (B, B) where the (i, j) entry is the Chamfer distance between
        batch1[i] and batch2[j].
    """
    if K is not None:
        batch1 = subsample_point_clouds(batch1, K)
        batch2 = subsample_point_clouds(batch2, K)
    
    B, N, _ = batch1.shape  # N is now either the original or the subsampled size K

    # Expand dimensions to pair up every combination of point clouds
    batch1_expanded = batch1.unsqueeze(1).expand(B, B, N, 3)  # shape: (B, B, N, 3)
    batch2_expanded = batch2.unsqueeze(0).expand(B, B, N, 3)  # shape: (B, B, N, 3)
    
    # Flatten the first two dimensions so that each pair is processed in one batched call.
    batch1_flat = batch1_expanded.reshape(B * B, N, 3)
    batch2_flat = batch2_expanded.reshape(B * B, N, 3)
    
    # Compute the Chamfer distance for all pairs at once.
    cd1, cd2, _, _ = chd(batch1_flat, batch2_flat)
    cd1, cd2 = cd1.mean(1), cd2.mean(1)  # Average over the points in each point cloud
    cd = cd1 + cd2  # Sum the distances from both directions
    # print("cd mean shape:", cd.shape)
    # print(cd.mean())
    # print(chamfer_distance(batch1_flat, batch2_flat))
    
    # Reshape the result into a (B, B) matrix.
    return cd.reshape(B, B)

def monte_carlo_iou(batch_a, batch_b=None, *, samples=200000, device="cuda"):
    """
    Approximate IoU matrix for two batches of 3-D SciPy ConvexHull objects
    using GPU-accelerated Monte-Carlo sampling.

    Returns a torch.Tensor of shape (len(batch_a), len(batch_b))
    on the requested device.
    """
    if batch_b is None:
        batch_b = batch_a
    N, M = len(batch_a), len(batch_b)

    # ---------------------------------------------------------------- helpers
    def hull_to_tensors(h):
        eq = h.equations.astype(np.float32)        # (n_f, 4)
        A = torch.as_tensor(eq[:, :3], device=device)   # (n_f, 3), float32
        d = torch.as_tensor(eq[:, 3],  device=device)   # (n_f,),  float32
        return A, d, float(h.volume)

    data_A = [hull_to_tensors(h) for h in batch_a]
    data_B = [hull_to_tensors(h) for h in batch_b]

    # ---------------------------------------------------------------- bounding box (float32!)
    all_pts = np.concatenate([h.points for h in set(batch_a + batch_b)])
    lo_np, hi_np = all_pts.min(axis=0), all_pts.max(axis=0)

    lo = torch.tensor(lo_np, dtype=torch.float32, device=device)
    hi = torch.tensor(hi_np, dtype=torch.float32, device=device)
    box_vol = float(np.prod(hi_np - lo_np))

    # ---------------------------------------------------------------- draw samples
    pts = lo + (hi - lo) * torch.rand(samples, 3, device=device)   # (S, 3) float32

    # ---------------------------------------------------------------- inside masks
    inside_A = torch.empty((N, samples), dtype=torch.bool, device=device)
    for i, (A, d, _) in enumerate(data_A):
        inside_A[i] = (A @ pts.T + d[:, None] <= 0).all(dim=0)

    inside_B = torch.empty((M, samples), dtype=torch.bool, device=device)
    for j, (A, d, _) in enumerate(data_B):
        inside_B[j] = (A @ pts.T + d[:, None] <= 0).all(dim=0)

    # ---------------------------------------------------------------- IoU matrix
    inter_counts = (inside_A[:, None, :] & inside_B[None, :, :]).sum(dim=-1)
    inter_vol = inter_counts.float() / samples * box_vol               # (N, M) float32

    vol_a = torch.tensor([v for _, _, v in data_A], dtype=torch.float32,
                         device=device).unsqueeze(1)                   # (N,1)
    vol_b = torch.tensor([v for _, _, v in data_B], dtype=torch.float32,
                         device=device).unsqueeze(0)                   # (1,M)

    union_vol = vol_a + vol_b - inter_vol
    return torch.where(union_vol > 0, inter_vol / union_vol,
                       torch.zeros_like(union_vol))

if __name__ == "__main__":
    # import numpy as np
    # # Example usage
    # K = 256  # Subsample to 50 points
    # all_contact_1 = np.load("/data/XXX/data/grasp_data/HumanHand/adiZero_Slide_2_SC/contact_pc/contact_pc_all.npy")
    # all_contact_2 = np.load("/data/XXX/data/grasp_data/HumanHand/adiZero_Slide_2_SC/contact_pc/contact_pc_all.npy")

    # for i in range(500):
    #     batch1 = torch.from_numpy(all_contact_1[i*256:(i+1)*256]).cuda()
    #     batch2 = torch.from_numpy(all_contact_2[i*256:(i+1)*256]).cuda()
    #     distance_matrix = compute_cross_chamfer_distance(batch1, batch2, K)
    #     print(distance_matrix.shape)

    import time
    rng = np.random.default_rng(0)
    for i in range(10):
        batch_A = [ConvexHull(rng.standard_normal((40, 3))) for _ in range(32)]
        batch_B = [ConvexHull(rng.standard_normal((35, 3)) + rng.uniform(-2.0, 2.0, 3))
                for _ in range(32)]
        curr = time.time()
        iou_gpu = monte_carlo_iou(batch_A, batch_B, samples=50000, device="cuda")
        print("GPU time:", time.time() - curr)
        print(iou_gpu.shape)
