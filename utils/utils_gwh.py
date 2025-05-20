import numpy as np
import trimesh
import igl
import torch
import math
from scipy.spatial import ConvexHull

def get_hull(hand_verts: np.ndarray,
             obj_mesh: trimesh.Trimesh,
             *,
             padding: float = 0.008,
             mu: float = 0.9,
             max_contact: int=128) -> ConvexHull | None:
    """
    Compute grasp-wrench convex hull for one grasp.

    Returns
    -------
    ConvexHull object, or None on failure / no contacts.
    """
    v, f = np.asarray(obj_mesh.vertices), np.asarray(obj_mesh.faces)
    sdf, _, _, N = igl.signed_distance(hand_verts, v, f, return_normals=True)

    pts = hand_verts[sdf <= padding]
    nrm = N[sdf <= padding]
    n = len(pts)
    if n > max_contact:
        # randomly sample MAX_CONTACTS points
        idx = np.random.choice(n, max_contact, replace=False)
        pts = pts[idx]
        nrm = nrm[idx]

    if len(pts) == 0:
        print(f"[skip] {obj_mesh}: no contact points")
        return None

    # two orthonormal tangents per normal
    k = nrm
    x = np.random.randn(len(pts), 3)
    x -= (x * k).sum(1)[:, None] * k
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.cross(k, x)

    w = np.zeros((len(pts), 4, 6))
    w[:, 0, :3] = k + mu * x
    w[:, 1, :3] = k - mu * x
    w[:, 2, :3] = k + mu * y
    w[:, 3, :3] = k - mu * y

    com   = obj_mesh.center_mass
    scale = np.abs(obj_mesh.bounds.T.reshape(6)).max()
    w[:,0,3:] = np.cross(pts - com, w[:,0,:3]) / scale
    w[:,1,3:] = np.cross(pts - com, w[:,1,:3]) / scale
    w[:,2,3:] = np.cross(pts - com, w[:,2,:3]) / scale
    w[:,3,3:] = np.cross(pts - com, w[:,3,:3]) / scale
    wrenches = w.reshape(-1, 6)
    # wrenches = w.reshape(-1, 3)
    wrenches = np.unique(wrenches, axis=0)

    # ───── filter out invalid rows ─────────────────────────────────────
    valid = np.isfinite(wrenches).all(axis=1)      # True where no NaN/Inf
    wrenches = wrenches[valid]

    if len(wrenches) < 9:      # convex hull needs at least 4 non‑coplanar pts
        return None

    try:
        hull = ConvexHull(wrenches, qhull_options="QJ QG6")# get_hull_safe(wrenches)
        return hull
    except Exception as e:
        # print(f"[skip] {obj_mesh}: {e}")
        # traceback.print_exc()
        return None

def monte_carlo_iou_1to1_6d(batch_a, batch_b, *,
                             samples: int = 1_000_000,
                             device: str = "cuda"):
    """
    Approximate 1-to-1 IoU for two equal-length lists of 6-D ConvexHulls
    using Monte-Carlo sampling on GPU.

    Parameters
    ----------
    batch_a, batch_b : Sequence[ConvexHull]
        Two lists of length N of SciPy ConvexHull objects in 6D.
    samples : int
        Number of random points to draw in the global 6D box.
    device : str
        PyTorch device ("cuda" or "cpu").

    Returns
    -------
    torch.Tensor of shape (N,)
        Estimated IoU for each i: IoU(batch_a[i], batch_b[i]).
    """
    assert len(batch_a) == len(batch_b), "batch sizes must match"
    N = len(batch_a)

    # 1) Precompute (A, d, vol) for each hull
    def hull_to_tensors(h):
        eq = h.equations.astype(np.float32)       # (n_faces, 7)
        A = torch.as_tensor(eq[:, :6], device=device)  # (n_faces, 6)
        d = torch.as_tensor(eq[:, 6],  device=device)  # (n_faces,)
        return A, d, float(h.volume)

    data_A = [hull_to_tensors(h) for h in batch_a]
    data_B = [hull_to_tensors(h) for h in batch_b]

    # 2) Compute global axis-aligned 6D bounding box
    all_pts = np.vstack([h.points for h in set(batch_a + batch_b)])
    lo_np, hi_np = all_pts.min(axis=0), all_pts.max(axis=0)
    lo = torch.tensor(lo_np, dtype=torch.float32, device=device)
    hi = torch.tensor(hi_np, dtype=torch.float32, device=device)
    box_vol = float(np.prod(hi_np - lo_np))

    # 3) Draw uniform samples in [lo, hi]^6
    pts = lo + (hi - lo) * torch.rand(samples, 6, device=device)  # (S,6), float32

    # 4) Build a boolean mask of shape (N, S) saying which pts lie inside each hull
    inside_A = torch.empty((N, samples), dtype=torch.bool, device=device)
    inside_B = torch.empty((N, samples), dtype=torch.bool, device=device)

    for i, (A, d, _) in enumerate(data_A):
        # test A @ x + d <= 0  for all faces
        inside_A[i] = (A @ pts.T + d[:, None] <= 0).all(dim=0)

    for i, (A, d, _) in enumerate(data_B):
        inside_B[i] = (A @ pts.T + d[:, None] <= 0).all(dim=0)

    # 5) For each i, count intersection, estimate volumes, and compute IoU
    #    counts_i = (#pts inside both A[i] and B[i])
    both = inside_A & inside_B       # (N, S)
    counts = both.sum(dim=1).float() # (N,)

    inter_vol = counts / samples * box_vol  # (N,)
    vol_a = torch.tensor([v for *_, v in data_A], device=device)  # (N,)
    vol_b = torch.tensor([v for *_, v in data_B], device=device)  # (N,)

    union_vol = vol_a + vol_b - inter_vol
    # avoid divide‑by‑zero
    iou = torch.where(union_vol > 0,
                      inter_vol / union_vol,
                      torch.zeros_like(union_vol))

    return iou

def monte_carlo_iou_1to1_6d_chunked(batch_a, batch_b, *,
                                     samples: int = 1_000_000,
                                     chunk_size: int = 100_000,
                                     device: str = "cuda"):
    """
    Same API as before, but processes the S Monte-Carlo points in chunks
    of at most `chunk_size` to bound peak GPU memory.
    """
    assert len(batch_a) == len(batch_b), "batch sizes must match"
    N = len(batch_a)

    # Precompute hull data on CPU, move to device
    def hull_to_tensors(h):
        eq = h.equations.astype(np.float32)    # (n_faces, 7)
        A = torch.as_tensor(eq[:, :6], device=device)
        d = torch.as_tensor(eq[:, 6], device=device)
        return A, d, float(h.volume)

    data_A = [hull_to_tensors(h) for h in batch_a]
    data_B = [hull_to_tensors(h) for h in batch_b]

    # Global 6D box
    all_pts = np.vstack([h.points for h in set(batch_a + batch_b)])
    lo_np, hi_np = all_pts.min(axis=0), all_pts.max(axis=0)
    lo = torch.tensor(lo_np, dtype=torch.float32, device=device)
    hi = torch.tensor(hi_np, dtype=torch.float32, device=device)
    box_vol = float(np.prod(hi_np - lo_np))

    # Prepare running intersection counts
    counts = torch.zeros(N, dtype=torch.long, device=device)
    drawn = 0

    # How many chunks?
    n_chunks = math.ceil(samples / chunk_size)

    for _ in range(n_chunks):
        this_chunk = min(chunk_size, samples - drawn)
        # (S_chunk, 6) float16 – you can drop .half() if precision is an issue
        pts = (lo + (hi - lo) * torch.rand(this_chunk, 6, device=device)).half()

        # build masks on the fly and accumulate
        # NOTE: bool mask of size (N, this_chunk)
        mask_A = torch.empty((N, this_chunk), dtype=torch.bool, device=device)
        for i, (A, d, _) in enumerate(data_A):
            # cast A, d to half if you want lower mem here too
            mask_A[i] = (A.half() @ pts.T + d.half()[:, None] <= 0).all(dim=0)

        mask_B = torch.empty((N, this_chunk), dtype=torch.bool, device=device)
        for i, (A, d, _) in enumerate(data_B):
            mask_B[i] = (A.half() @ pts.T + d.half()[:, None] <= 0).all(dim=0)

        # intersection counts per hull pair
        counts += (mask_A & mask_B).sum(dim=1)

        # book‑keeping and free peak
        drawn += this_chunk
        del pts, mask_A, mask_B
        torch.cuda.empty_cache()

    # final IoU
    inter_vol = counts.float() / samples * box_vol   # (N,)
    vol_a = torch.tensor([v for *_, v in data_A], device=device)
    vol_b = torch.tensor([v for *_, v in data_B], device=device)
    union_vol = vol_a + vol_b - inter_vol

    return torch.where(union_vol > 0,
                       inter_vol / union_vol,
                       torch.zeros_like(union_vol))

def simplify_mesh(mesh: trimesh.Trimesh, *,
                target_faces: int = 5000,
                max_iter: int = 30) -> trimesh.Trimesh:
    """
    Simplify a mesh to a target number of faces using quadric decimation.
    """
    for _ in range(max_iter):
        mesh = mesh.simplify_quadric_decimation(face_count=int(len(mesh.faces) * 0.7))
        mesh.remove_unreferenced_vertices()
        mesh.remove_duplicate_faces()
        if len(mesh.faces) < target_faces:
            break
    if len(mesh.faces) > target_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
        mesh.remove_unreferenced_vertices()
        mesh.remove_duplicate_faces()
    return mesh

# -------------------- example usage --------------------
if __name__ == "__main__":
    import time
    for i in range(100):
        rng = np.random.default_rng(42)
        N = 64
        # Create two batches of 6D hulls (random clouds)
        batch_A = [ConvexHull(rng.standard_normal((50, 6))) for _ in range(N)]
        batch_B = [ConvexHull(rng.standard_normal((50, 6)) + rng.uniform(-0.3,0.3,6))
                for _ in range(N)]

        iou_6d = monte_carlo_iou_1to1_6d_chunked(batch_A, batch_B,
                                        samples=500_000,
                                        device="cuda")
        print(iou_6d.shape)  # torch.Size([64])
        print(iou_6d[:5])    # first 5 estimates
    # hand_verts = np.load("/data/XXX/data/grasp_data/Allegro/003_cracker_box/hand_pc/allegro_hand_description_right_pc_0.npy")
    # hand_verts = hand_verts[:, :3]
    # obj_mesh = trimesh.load("/data/XXX/data/mgg_pc/objects/obj/003_cracker_box.obj")
    # for _ in range(30):
    #     obj_mesh = obj_mesh.simplify_quadric_decimation(face_count=int(len(obj_mesh.faces) * 0.7))
    #     obj_mesh.remove_unreferenced_vertices()
    #     obj_mesh.remove_duplicate_faces()
    #     if len(obj_mesh.faces) < 5000:
    #         break
    # if len(obj_mesh.faces) > 5000:
    #     obj_mesh = obj_mesh.simplify_quadric_decimation(face_count=5000)
    # obj_mesh.remove_unreferenced_vertices()
    # obj_mesh.remove_duplicate_faces()
    # curr = time.time()
    # hull = get_hull(hand_verts, obj_mesh, padding=0.008, mu=0.9)
    # print("Time to compute hull:", time.time() - curr)
