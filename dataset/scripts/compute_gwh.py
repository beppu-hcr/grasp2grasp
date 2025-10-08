"""
Compute and save grasp wrench hulls (GWHs) for an entire dataset.

Fast I/O version:
-----------------
* Each <hand_X/object_Y>/ folder will contain exactly one file:  gwh.npy
  • dtype = object
  • element i is either None or a scipy.spatial.ConvexHull instance

Example
-------
python compute_gwh.py --root grasp_data --num-workers 16

Later in your training code
---------------------------
from compute_gwh import load_gwh
gwhs = load_gwh("grasp_data/hand_1/object_1/gwh.npy", mmap=True)
batch = gwhs[batch_idx]        # list/array of ConvexHull | None
"""

from __future__ import annotations
import argparse, warnings, traceback, os
from pathlib import Path
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import multiprocessing as mp
from contextlib import closing
from natsort import natsorted

import numpy as np
# import pandas as pd
# import torch
# from pytorch3d.transforms import matrix_to_rotation_6d, quaternion_to_matrix
import trimesh
import igl                                     # libigl python bindings
from scipy.spatial import ConvexHull
from tqdm import tqdm
# from grippers.HandLayer import grabHandLayer

TARGET_FACES = 5000
MAX_CONTACTS = 64
# BATCH_SIZE = 1024
# ────────────────────────── Hull computation ──────────────────────────
def get_hull(hand_verts: np.ndarray,
             obj_mesh: trimesh.Trimesh,
             *,
             padding: float = 0.008,
             mu: float = 0.9) -> ConvexHull | None:
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
    if n > MAX_CONTACTS:
        # randomly sample MAX_CONTACTS points
        idx = np.random.choice(n, MAX_CONTACTS, replace=False)
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

    w = np.zeros((len(pts), 4, 3))
    w[:, 0, :3] = k + mu * x
    w[:, 1, :3] = k - mu * x
    w[:, 2, :3] = k + mu * y
    w[:, 3, :3] = k - mu * y

    # com   = obj_mesh.center_mass
    # scale = np.abs(obj_mesh.bounds.T.reshape(6)).max()
    # w[:,0,3:] = np.cross(pts - com, w[:,0,:3]) / scale
    # w[:,1,3:] = np.cross(pts - com, w[:,1,:3]) / scale
    # w[:,2,3:] = np.cross(pts - com, w[:,2,:3]) / scale
    # w[:,3,3:] = np.cross(pts - com, w[:,3,:3]) / scale
    # wrenches = w.reshape(-1, 6)
    wrenches = w.reshape(-1, 3)
    wrenches = np.unique(wrenches, axis=0)

    # ───── filter out invalid rows ─────────────────────────────────────
    valid = np.isfinite(wrenches).all(axis=1)      # True where no NaN/Inf
    wrenches = wrenches[valid]

    if len(wrenches) < 6:      # convex hull needs at least 4 non‑coplanar pts
        return None

    try:
        hull = ConvexHull(wrenches, qhull_options="QJ QG6")# get_hull_safe(wrenches)
        return hull
    except Exception as e:
        print(f"[skip] {obj_mesh}: {e}")
        traceback.print_exc()
        return None

# ---------------------------------------------------------------
# low-level target executed in *another* process
# ---------------------------------------------------------------
def _safe_qhull(wrenches, options, conn):
    try:
        hull = ConvexHull(wrenches, qhull_options=options)
        conn.send(hull)        # pickles the ConvexHull object
    except Exception as e:     # numeric failure, but no segfault
        print(f"[skip] {e}")
        conn.send(e)
    finally:
        conn.close()


def get_hull_safe(wrenches: np.ndarray,
                  *,
                  qhull_opts: str = "QJ QG6",
                  timeout: float = 10.0) -> ConvexHull | None:
    """
    Wrap ConvexHull in a dedicated process to survive SIGSEGV.

    Returns
    -------
    ConvexHull on success, None on any failure (numeric or crash).
    """
    parent, child = mp.Pipe(duplex=False)
    proc = mp.get_context("spawn").Process(target=_safe_qhull,
                                           args=(wrenches, qhull_opts, child))
    proc.start()
    child.close()              # parent only needs its own end

    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()

    if proc.exitcode == 0 and parent.poll():
        result = parent.recv()
        if isinstance(result, ConvexHull):
            return result
    # any non-zero exitcode (incl. segfault = -11) or other issue ⇒ failure
    return None


# ────────────────────────── Fast I/O helpers ──────────────────────────
def _save_hulls_npy(dst_file: Path, hulls: list[ConvexHull | None]) -> None:
    """Save list of hulls to uncompressed .npy (pickled object array)."""
    np.save(dst_file, np.asarray(hulls, dtype=object), allow_pickle=True)


def load_gwh(path: str | os.PathLike,
             *,
             mmap: bool | str | None = False) -> np.ndarray:
    """
    Load gwh.npy and return an object-dtype array whose elements are
    ConvexHull or None.

    Parameters
    ----------
    path  : '.../gwh.npy'
    mmap  : False | True | 'r' | 'r+' - pass-through to np.load(mmap_mode=...)
            True -> read-only mmap ('r')

    Returns
    -------
    ndarray(dtype=object)
    """
    mode = 'r' if mmap is True else mmap
    return np.load(path, allow_pickle=True, mmap_mode=mode)


# ────────────────────────── Worker for multiprocessing ─────────────────
def _worker(args):
    """Return (index, hull_or_none).  Keeps order when using imap_unordered."""
    idx, hand_pc_path, obj_mesh, padding = args
    hand_pc  = np.load(hand_pc_path)
    hand_pc = hand_pc[:, :3]
    # obj_mesh = trimesh.load_mesh(obj_mesh_path, process=False)
    hull     = get_hull(hand_pc, obj_mesh, padding=padding)
    return idx, hull


# ────────────────────────── Per-object handler ────────────────────────
    
def _handle_object(obj_dir: Path,
                   # hand_layer,
                   *,
                   padding: float,
                   num_workers: int):
    """
    Compute all hulls inside one <hand_X/object_Y>/ directory and write gwh.npy
    """
    if (obj_dir / "gwh.npy").exists():
        print(f"[skip] {obj_dir}: gwh.npy already exists")
        return

    # obj meshes are in "/data/XXX/data/mgg_pc/objects/obj/<obj_id>.obj"
    obj_mesh_path = Path("/data/XXX/data/mgg_pc/objects/obj") / f"{obj_dir.name}.obj"
    if not obj_mesh_path.exists():
        warnings.warn(f"[skip] {obj_dir}: obj mesh not found")
        return
    obj_mesh = trimesh.load_mesh(obj_mesh_path, process=False)
    for _ in range(30):
        obj_mesh = obj_mesh.simplify_quadric_decimation(face_count=int(len(obj_mesh.faces) * 0.7))
        obj_mesh.remove_unreferenced_vertices()
        obj_mesh.remove_duplicate_faces()
        if len(obj_mesh.faces) < TARGET_FACES:
            break
    if len(obj_mesh.faces) > TARGET_FACES:
        obj_mesh = obj_mesh.simplify_quadric_decimation(face_count=TARGET_FACES)
    obj_mesh.remove_unreferenced_vertices()
    obj_mesh.remove_duplicate_faces()

    hand_pc_dir = obj_dir / "hand_pc"
    pcs = natsorted(hand_pc_dir.glob("*.npy"))
    if not pcs:
        warnings.warn(f"[skip] {obj_dir}: no hand pcs")
        return
    # Get string path
    # obj_dir_str = str(obj_dir)
    # pose_all, theta_all = extract_posetheta_from_metadata(obj_dir_str)
    # pose_tensor = torch.tensor(pose_all)
    # theta_tensor = torch.tensor(theta_all)
    
    # hand_verts_all = []
    # for i in range(len(pose_tensor) // BATCH_SIZE + 1):
    #     batch_pose = pose_tensor[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
    #     batch_theta = theta_tensor[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
    #     hand_verts = hand_layer(batch_pose, batch_theta)
    #     hand_verts_all.append(hand_verts.cpu().numpy())
    # hand_verts_all = np.concatenate(hand_verts_all, axis=0)

    # Pre-allocate list for ordering
    hulls: list[ConvexHull | None] = [None] * len(pcs)
    hulls_success = np.zeros(len(pcs), dtype=bool)
    # hulls_3d: list[ConvexHull | None] = [None] * len(pcs)
    args = [(i, pc, obj_mesh, padding) for i, pc in enumerate(pcs)]

    with ThreadPool(num_workers) as pool, tqdm(total=len(pcs),
                                         desc=str(obj_dir.relative_to(obj_dir.parents[2])),
                                         leave=False) as bar:
        for i, hull in pool.imap_unordered(_worker, args):
            hulls[i] = hull
            hulls_success[i] = hull is not None
            # hulls_3d[i] = hull[1]
            bar.update()

    _save_hulls_npy(obj_dir / "gwh.npy", hulls)
    np.save(obj_dir / "hulls_success.npy", hulls_success)
    # _save_hulls_npy(obj_dir / "gwh_3d.npy", hulls_3d)


# ────────────────────────── Dataset traversal ─────────────────────────
def _traverse(root: Path, 
              hand: str,
                *,
              padding: float,
              num_workers: int):
    hand_dir = root / hand
    # hand_layer = grabHandLayer(hand, device='cuda:0')
    for obj_dir in sorted(hand_dir.glob("*")):
        if not obj_dir.is_dir():
            continue
        _handle_object(obj_dir,
                       # hand_layer,
                        padding=padding,
                        num_workers=num_workers)


# ────────────────────────── CLI entry-point ───────────────────────────
def main():
    ap = argparse.ArgumentParser("GWH pre-computation (fast I/O version)")
    ap.add_argument("--root", required=True, type=str,
                    default='../../data/grasp_data',
                    help="root dataset folder (e.g. grasp_data)")
    ap.add_argument("-p", "--padding", type=float, default=0.008,
                    help="contact threshold (m)")
    ap.add_argument("-j", "--num-workers", type=int, default=os.cpu_count(),
                    help="multiprocessing workers (default: all cores)")
    args = ap.parse_args()

    for hand in ["Allegro", "HumanHand", "shadow_hand"]:
        _traverse(Path(args.root).expanduser(),
                    hand=hand,
                padding=args.padding,
                num_workers=args.num_workers)


if __name__ == "__main__":
    main()
