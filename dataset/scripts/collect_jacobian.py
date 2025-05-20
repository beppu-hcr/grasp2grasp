import builtins

# stash original
_orig_print = builtins.print

def _filtered_print(*args, **kwargs):
    # join all args into a single string
    msg = " ".join(str(a) for a in args)
    # drop lines like "Module ___ load took ___ ms"
    if msg.startswith("Module ") and " load took " in msg:
        return
    # otherwise, print as normal
    _orig_print(*args, **kwargs)

# override the builtin
builtins.print = _filtered_print
import os, sys, contextlib
import argparse
import numpy as np
import warp as wp
import warp.sim
import warp.sim.render
import warp.render
from pathlib import Path
import trimesh as tri
from scipy.spatial.transform import Rotation
import pandas as pd
import time
from tqdm import tqdm

fnull = open(os.devnull, 'w')
with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
    wp.init()

# restore normal output
fnull.close()

# -- Warp utility functions -------------------------------------------------- #
@wp.func
def project(p: wp.vec3, c: wp.vec3, n: wp.vec3):
    return p - wp.dot(p - c, n) * n

@wp.func
def quat_mul(a: wp.quat, b: wp.quat) -> wp.quat:
    # quaternion components in (x,y,z,w) ordering
    ax = a[0]
    ay = a[1]
    az = a[2]
    aw = a[3]

    bx = b[0]
    by = b[1]
    bz = b[2]
    bw = b[3]

    # (w, x, y, z) Hamilton product
    rw = aw*bw - ax*bx - ay*by - az*bz
    rx = aw*bx + ax*bw + ay*bz - az*by
    ry = aw*by - ax*bz + ay*bw + az*bx
    rz = aw*bz + ax*by - ay*bx + az*bw

    return wp.quat(rx, ry, rz, rw)

@wp.kernel
def extract_loss(obj_q: wp.array(dtype=wp.vec3, ndim=2), loss: wp.array(dtype=float, ndim=1)):
    tid = wp.tid()
    if tid == 0:
        # take the x-component of the first batch
        loss[0] = obj_q[0,0][0]

@wp.kernel
def extract_scalar(
    obj_q:   wp.array(dtype=wp.vec3, ndim=2),
    obj_rot: wp.array(dtype=wp.quat, ndim=2),
    idx:     wp.int32,
    loss:    wp.array(dtype=float, ndim=1)):
    if wp.tid() == 0:
        if idx < 3:
            loss[0] = obj_q[0,0][idx]
        else:
            loss[0] = obj_rot[0,0][idx-3]

# Zero out object state
@wp.kernel
def zero_out(
    obj_q: wp.array(dtype=wp.vec3, ndim=2),        # [N_BATCH x 1]
    obj_qd: wp.array(dtype=wp.vec3, ndim=2),       # [N_BATCH x 1]
    obj_rot: wp.array(dtype=wp.quat, ndim=2),      # [N_BATCH x 1]
    N_BATCH: wp.int32):
    tid = wp.tid()
    # Only one direction per batch for Jacobian
    obj_q[tid, 0] = wp.vec3(0.0)
    obj_qd[tid, 0] = wp.vec3(0.0)
    # identity quaternion
    obj_rot[tid, 0] = wp.quat(0.0, 0.0, 0.0, 1.0)

# Simulation step: update translation and rotation via penalty-based contact
@wp.kernel
def sim_step(
    obj_mesh_id: wp.uint64,
    shape_mesh_ids: wp.array(dtype=wp.uint64, ndim=1),
    obj_mesh_normals: wp.array(dtype=wp.vec3, ndim=1),
    body_q: wp.array(dtype=wp.transform, ndim=1),
    contact_shape1: wp.array(dtype=wp.int32, ndim=1),
    contact_body0: wp.array(dtype=wp.int32, ndim=1),
    contact_body1: wp.array(dtype=wp.int32, ndim=1),
    contact_point0: wp.array(dtype=wp.vec3, ndim=1),
    shape_tf: wp.array(dtype=wp.transform, ndim=1),
    dt: float,
    padding: wp.array(dtype=float, ndim=1),
    N_BATCH: wp.int32,
    N_CONTACTS: wp.int32,
    N_SHAPES: wp.int32,
    obj_q: wp.array(dtype=wp.vec3, ndim=2),
    obj_qd: wp.array(dtype=wp.vec3, ndim=2),
    obj_rot: wp.array(dtype=wp.quat, ndim=2)):
    tid = wp.tid()
    n_batch = tid // N_CONTACTS
    n_contact = tid % N_CONTACTS

    # world contact point
    x_world = wp.transform_point(body_q[contact_body0[n_contact]], contact_point0[n_contact])

    # unconstrained motion
    obj_qd_unconstrained = wp.vec3(0.0)
    obj_q_unconstrained = wp.vec3(0.0)

    # mesh query (updated API)
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    max_dist = 1e8
    wp.mesh_query_point(obj_mesh_id, x_world, max_dist, sign, face_index, face_u, face_v)

    # barycentric weight
    w = 1.0 - face_u - face_v
    p0 = wp.mesh_get_point(obj_mesh_id, face_index * 3)
    p1 = wp.mesh_get_point(obj_mesh_id, face_index * 3 + 1)
    p2 = wp.mesh_get_point(obj_mesh_id, face_index * 3 + 2)
    n0 = obj_mesh_normals[wp.mesh_get_index(obj_mesh_id, face_index * 3)]
    n1 = obj_mesh_normals[wp.mesh_get_index(obj_mesh_id, face_index * 3 + 1)]
    n2 = obj_mesh_normals[wp.mesh_get_index(obj_mesh_id, face_index * 3 + 2)]
    p = face_u * p0 + face_v * p1 + w * p2
    c0 = project(p, p0, n0)
    c1 = project(p, p1, n1)
    c2 = project(p, p2, n2)
    q = face_u * c0 + face_v * c1 + w * c2

    # compute signed penetration via normal
    interpolated_n = face_u * n0 + face_v * n1 + w * n2
    dotp = wp.dot(interpolated_n, x_world - q)
    sign = 1.0
    if dotp < 0.0:
        sign = -1.0
    penetration = -dotp - padding[0] + 0.001
    C = wp.max(0.0, penetration)

    # impulse calculation
    n_dir_vec = wp.normalize(x_world - q) * sign
    eps = 1e-6
    lmbda = -C / (wp.length(n_dir_vec) * wp.length(n_dir_vec) + eps)
    impulse = lmbda * n_dir_vec / float(N_CONTACTS)

    if n_contact == 0:
        # translation update
        wp.atomic_add(obj_q, n_batch, 0, obj_q_unconstrained)
        wp.atomic_add(obj_qd, n_batch, 0, obj_qd_unconstrained)
    # rotation via torque = r x impulse
    wp.atomic_add(obj_q,  n_batch, 0, impulse * dt * dt)
    wp.atomic_add(obj_qd, n_batch, 0, impulse * dt)
    # rotation update
    torque = wp.cross(x_world, impulse)
    I = 500.0
    omega = torque / I
    omega_q = wp.quat(omega[0], omega[1], omega[2], 0.0)
    prev_q = obj_rot[n_batch, 0]
    q_dot = 0.5 * quat_mul(prev_q, omega_q)
    q_new = prev_q + q_dot * dt
    obj_rot[n_batch, 0] = wp.normalize(q_new)
    
    # ang_vel = wp.length(omega)
    # axis = wp.normalize(omega + wp.vec3(eps))
    # max_step = 0.1
    # delta_ang = wp.min(max_step, ang_vel * dt)
    # dq = wp.utils.quat_from_axis_angle(axis, delta_ang)

    # prev_q = obj_rot[n_batch, 0]
    # new_q = quat_mul(dq, prev_q)
    # obj_rot[n_batch, 0] = wp.normalize(new_q)
    # axis = wp.normalize(torque)
    # angle = wp.length(torque)
    # dq = wp.utils.quat_from_axis_angle(axis, angle)
    # wp.atomic_add(obj_rot, n_batch, 0, dq)
    # prev_q = obj_rot[n_batch, 0]
    # obj_rot[n_batch, 0] = wp.normalize(prev_q)

# ----------------------------------------------------------------------------- #

def compute_jacobian_for_grasp(model, state, obj_arrays, mesh_data, tape, graph):
    """
    Given a built model, state, and captured graph, compute Jacobian for a single grasp.
    """
    # Run forward+backward
    tape.zero()
    wp.synchronize()
    wp.capture_launch(graph)
    # Extract gradients: d(obj_q)/d(joint_q) and d(obj_rot)/d(joint_q)
    dq_grad = tape.gradients[model.joint_q].numpy()  # shape [N_JOINTS]
    # Note: obj_rot gradient also stored in tape.gradients ?
    return dq_grad

@wp.kernel
def set_render_model(
    sim_q:    wp.array(dtype=float, ndim=1),  # N_BATCH*N_JOINTS
    batch_id: wp.int32,                       # which batch index to copy
    N_JOINTS: wp.int32,
    rend_q:   wp.array(dtype=float, ndim=1)): # N_JOINTS
    tid = wp.tid()
    offset = batch_id * N_JOINTS
    rend_q[tid] = sim_q[offset + tid]


def main():
    parser = argparse.ArgumentParser(description="Compute Jacobian of object pose w.r.t. hand joints")
    parser.add_argument('--hand_urdf', type=str, required=True)
    parser.add_argument('--object_obj', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--dt', type=float, default=1e-2)
    args = parser.parse_args()

    # Build hand model
    builder = wp.sim.ModelBuilder()
    wp.sim.parse_urdf(args.hand_urdf, builder,
                      xform=wp.transform_identity(), # wp.transform(np.zeros(3), wp.quat_from_axis_angle(wp.vec3(1.0,0.0,0.0), -np.pi*0.5)),
                      floating=True,
                      density=0.1,
                      armature=0.1,
                      stiffness=0.0,
                      damping=0.0,
                      shape_ke=1.e+4,
                      shape_kd=1.e+2,
                      shape_kf=1.e+2,
                      shape_mu=1.0,
                      limit_ke=1.e+4,
                      limit_kd=1.e+1)
    model = builder.finalize("cuda")
    model.ground = False
    model.joint_attach_ke = 1600.0
    model.joint_attach_kd = 20.0
    state = model.state()
    model.collide(state)
    model.joint_q.requires_grad = True
    state.body_q.requires_grad = True
    # print(model.joint_q)

    # Load object mesh
    obj_mesh_tri = tri.load_mesh(args.object_obj)
    obj_mesh_verts = wp.array(shape=(len(obj_mesh_tri.vertices),), data=obj_mesh_tri.vertices,
                              dtype=wp.vec3, device="cuda")
    obj_mesh_inds = wp.array(shape=(obj_mesh_tri.faces.flatten().shape[0],), data=obj_mesh_tri.faces.flatten(),
                              dtype=int, device="cuda")
    obj_mesh_normals = wp.array(shape=(len(obj_mesh_tri.vertices),), data=obj_mesh_tri.vertex_normals,
                                dtype=wp.vec3, device="cuda")
    obj_mesh = wp.Mesh(points=obj_mesh_verts, indices=obj_mesh_inds)
    obj_mesh.refit()
    obj_normals = wp.array(shape=(len(obj_mesh_tri.vertices),), data=obj_mesh_tri.vertex_normals,
                           dtype=wp.vec3, device='cuda')

    # build shape meshes & IDs
    shape_ids = []
    for geom in model.shape_geo_src:
        if geom is None:
            continue
        v = wp.array(shape=(len(geom.vertices),), data=np.array(geom.vertices),
                     dtype=wp.vec3, device='cuda')
        f = wp.array(shape=(len(geom.indices),), data=np.array(geom.indices),
                     dtype=int, device='cuda')
        m = wp.Mesh(points=v, indices=f)
        m.refit()
        shape_ids.append(m.id)
    shape_ids_np = np.array(shape_ids, dtype=np.uint64)
    shape_mesh_ids = wp.array(shape=(len(shape_ids),), data=shape_ids_np,
                               dtype=wp.uint64, device='cuda')


    # Allocate state arrays
    N_BATCH = args.batch_size
    N_CONTACTS = len(model.contact_shape0) // N_BATCH
    # print(f"Number of contacts: {N_CONTACTS}")
    # Single direction per grasp
    obj_q   = wp.zeros((N_BATCH,1), dtype=wp.vec3, device="cuda", requires_grad=True)
    obj_qd  = wp.zeros((N_BATCH,1), dtype=wp.vec3, device="cuda", requires_grad=True)
    obj_rot = wp.zeros((N_BATCH,1), dtype=wp.quat, device="cuda", requires_grad=True)
    padding = wp.zeros((1,), dtype=float, device="cuda")
    loss    = wp.zeros((1,),       dtype=float, device='cuda', requires_grad=True)

    # Capture graph (forward + backward)
    graphs = []
    tapes  = []
    for i in range(7):
        tape = wp.Tape()
        wp.capture_begin()
        with tape:
            wp.launch(kernel=zero_out, dim=N_BATCH, inputs=[obj_q, obj_qd, obj_rot, N_BATCH], device="cuda")
            wp.sim.eval_fk(model, model.joint_q, model.joint_qd, None, state)
            wp.launch(kernel=sim_step,
                    dim=N_BATCH * N_CONTACTS,
                    inputs=[
                        obj_mesh.id,
                        shape_mesh_ids,
                        obj_mesh_normals,
                        state.body_q,
                        model.contact_shape1,
                        model.contact_body0,
                        model.contact_body1,
                        model.contact_point0,
                        model.shape_transform,
                        args.dt,
                        padding,
                        N_BATCH,
                        N_CONTACTS,
                        len(model.shape_geo_src),
                        obj_q,
                        obj_qd,
                        obj_rot],
                    device="cuda")
            wp.launch(extract_scalar, dim=1, inputs=[obj_q, obj_rot, i, loss], device='cuda')
            # Backward on each quaternion component or translation as needed
            # Here, as an example, we backprop on obj_q.x
        tape.backward(loss)
        graph = wp.capture_end()
        tapes.append(tape)
        graphs.append(graph)

    # Example: compute Jacobian for one grasp (loop over dataset similarly)
    # Load a single grasp q vector here...
    hand_name = os.path.dirname(args.hand_urdf).split('/')[-1]
    object_name = os.path.basename(args.object_obj).split('.')[0]
    df = pd.read_parquet(f"/data/XXX/data/grasp_data/{hand_name}/{object_name}/metadata.parquet")
    poses = np.stack(df['hand_pose'].tolist())
    thetas = np.stack(df['graspit_dofs'].tolist())

    poses = poses[:, [0, 1, 2, 4, 5, 6, 3]]
    if "shadow" in args.hand_urdf:
        # print("Shadow hand detected")
        # print(model.joint_q.shape)
        # print(thetas.shape)
        thetas = thetas[:, [17, 18, 19, 20, 21, 0, 1, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16, 4, 5, 6, 7, 8]]

    J_max = np.zeros((poses.shape[0],7), dtype=np.float32)
    J_success = np.zeros((poses.shape[0],), dtype=bool)
    for i in range(poses.shape[0]):
        grasp_q = np.concatenate([poses[i], thetas[i]])
        # grasp_q = np.zeros(7 + len(thetas[i]), dtype=np.float32)
        # grasp_q[6] = 1.0
        # print(grasp_q)
        model.joint_q.assign(grasp_q)
        # print(model.joint_q.numpy())

        # Run graph to get gradients
        #curr = time.time()
        for dim in range(7):
            tapes[dim].zero()
            wp.synchronize()
            wp.capture_launch(graphs[dim])
            # print(obj_q.numpy())
            # wp.launch(extract_loss, dim=1, inputs=[obj_q, loss], device='cuda')
            # tape.backward(loss)
            # print(model.joint_q.numpy())
            # print(tape.gradients.keys())
            jacobian = tape.gradients[model.joint_q].numpy()  # shape [N_JOINTS]
            # print(f"Jacobian shape: {jacobian.shape}")
            # print(f"Jacobian: {jacobian}")
            joint_jacobian = jacobian[7:]
            if np.all(np.isnan(joint_jacobian)):
                J_success[i] = False
                J_max[i, dim] = 0
            else:
                joint_jacobian = np.nan_to_num(joint_jacobian)
                J_max[i, dim] = np.max(np.abs(joint_jacobian))
                J_success[i] = True
    jac_save_path = os.path.join("/data/XXX/data/grasp_data", hand_name, object_name, "jacobian.npy")
    suc_save_path = os.path.join("/data/XXX/data/grasp_data", hand_name, object_name, "jacobian_success.npy")
    np.save(jac_save_path, J_max)
    np.save(suc_save_path, J_success)
    #print(f"Time taken: {time.time() - curr} seconds")

    # Rendering
    # sim_builder = wp.sim.ModelBuilder()
    # wp.sim.parse_urdf(
    #     args.hand_urdf,
    #     sim_builder,
    #     xform=wp.transform_identity(),
    #     floating=True,
    # )
    # render_model = sim_builder.finalize("cuda")
    # render_model.ground = False
    # render_state = render_model.state()

    # # 2) Copy the joint angles from your sim model into the render model
    # sim_joint_q = np.concatenate([poses[0], thetas[0]])
    # N_JOINTS = len(sim_joint_q)
    # sim_q_flat = sim_joint_q.astype(np.float32).flatten()
    # # broadcast to batch=1
    # sim_q_buff = wp.array(shape=(1*N_JOINTS,), data=sim_q_flat, dtype=float, device="cuda")
    # wp.launch(
    #     set_render_model,
    #     dim=N_JOINTS,
    #     inputs=[sim_q_buff, 0, N_JOINTS,
    #             render_model.joint_q],
    #     device="cuda"
    # )
    # wp.sim.eval_fk(
    #     render_model, render_model.joint_q, render_model.joint_qd,
    #     None, render_state
    # )

    # # 4) Create the USD renderer and dump one frame
    # r = wp.sim.render.SimRenderer(render_model, "debug.usd")

    # t = 0.0
    # r.begin_frame(t)
    # # renders the hand
    # r.render(render_state)
    # # overlays the object mesh in its local origin
    # r.render_mesh(
    #     name="object",
    #     points=obj_mesh.points.numpy(),
    #     indices=obj_mesh.indices.numpy()
    # )
    # r.end_frame()

    # r.save()

    # Save
    # os.makedirs(args.output_dir, exist_ok=True)
    # np.savetxt(os.path.join(args.output_dir, 'jacobian.txt'), jacobian)
    # print(f"Jacobian saved to {args.output_dir}/jacobian.txt")

if __name__ == '__main__':
    main()
