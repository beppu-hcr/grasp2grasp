from copy import deepcopy
from glob import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.io as sio
from scipy.spatial.transform import Rotation as R
import shutil
import trimesh
from urdfpy.urdfpy.urdf import URDF

from contact_utils import *


def main():
    hands = {
                # "Allegro": ["allegro_hand_description_right"], 
                # "HumanHand": ["HumanHand"],
                "shadow_hand": ["shadow_hand"]
    }
    urdf_base_path = "/workspace/code/Point-VAE/isaac_sim_grasping/grippers"
    graspit_base_path = "/diskstation/XXX/data/multigripper_grasp_data/Dataset/graspit_grasps"
    output_path = "/diskstation/XXX/data/test"
    data_info_columns = ['hand', 'object_id', 'pc_path', 'initial_dofs', 'final_dofs', 'graspit_dofs', 'hand_pose']
    data_info = []

    for hand, hand_urdf_names in hands.items():
        Path.mkdir(Path(f"{output_path}/{hand}"), parents=True, exist_ok=True)

        for hand_urdf_name in hand_urdf_names:
            data_output_path = f"{output_path}/{hand}"
            Path.mkdir(Path(data_output_path), exist_ok=True)

            urdf_path = f"{urdf_base_path}/{hand}"
            robot = URDF.load(f'{urdf_path}/{hand_urdf_name}.urdf')
            # actuated_joints = [joint for joint in robot.joints if 'tip' not in joint.name] 
            actuated_joints = robot.actuated_joints
            # actuated_joints = [joint for joint in robot.joints if joint.joint_type != "fixed"]
            # actuated_joints = sorted(actuated_joints, key=lambda joint: joint.name)            
            # for t in actuated_joints:
            #     print(t.name)

            graspit_paths = glob(f'{graspit_base_path}/{hand}/*.json')
            for graspit_path in graspit_paths:
                with open(graspit_path, 'r') as file:
                    graspit_data = json.load(file)

                object_id = graspit_data['object_id']

                output_path = f"{data_output_path}/{object_id}"
                pc_path = f"{output_path}/hand_pc"
                contact_path = f"{output_path}/contact_pc"
                hand_obj_path = f"{output_path}/hand_obj"
                object_path = f"{output_path}/{object_id}.npy"
                Path.mkdir(Path(output_path), exist_ok=True)
                Path.mkdir(Path(pc_path), exist_ok=True)
                Path.mkdir(Path(contact_path), exist_ok=True)
                Path.mkdir(Path(hand_obj_path), exist_ok=True)

                # Temp: Objects already generated. Copy them here
                base_object_path = "/diskstation/XXX/data/mgg_pc/objects/npy"
                src_object_path = f"{base_object_path}/{object_id}.npy"
                object_path = f"{output_path}/{object_id}.npy"
                shutil.copy(src_object_path, object_path)

                # Read Object Point Cloud (XYZ)
                object_points = np.load(object_path)
                threshold = 0.01
                kd_tree, obj_pcd = build_kdtree_for_object(object_points) # shape: (N,3)

                data_info = []
                for grasp_idx in range(len(graspit_data['pose'])):
                    # Hand Pose
                    hand_pose = graspit_data['pose'][grasp_idx]
                    hand_T = pose_to_matrix(hand_pose)

                    # Link poses
                    joint_mapping = {} 
                    for i, joint in enumerate(actuated_joints):
                        joint_mapping[joint] = graspit_data['graspit_dofs'][grasp_idx][i]
                    link_poses = robot.link_fk(cfg=joint_mapping)

                    # Sample mesh points to generate pc
                    scene = []
                    for link in robot.links:
                        # Transform link to final dof
                        link_pose_global = hand_T @ link_poses[link]
                        for visual in link.visuals:
                            if visual.geometry.mesh is not None:
                                # Load mesh
                                mesh_file = visual.geometry.mesh.filename
                                mesh_file = f'{urdf_path}/{mesh_file[2:]}'
                                mesh = trimesh.load(mesh_file)

                                # Transform mesh to global frame
                                mesh_tf = link_pose_global @ visual.origin
                                mesh.apply_transform(mesh_tf)
                                scene.append(mesh)

                    # Sample Mesh Points
                    hand_mesh = trimesh.util.concatenate(scene)
                    points, _ = trimesh.sample.sample_surface(hand_mesh, 4096)  
                    points = np.vstack(points)

                    # Save Data
                    if points.shape[0] > 0:
                        # Generate Contact Points
                        contact_points_dt = find_object_contact_points_kdtree(kd_tree,
                                                                            obj_pcd,
                                                                            points,
                                                                            threshold)
            
                        # Hand Class flag
                        ones_col = np.zeros((points.shape[0], 1))
                        points = np.hstack([points, ones_col])

                        np.save(f"{pc_path}/{hand}_pc_{grasp_idx}.npy", points)
                        hand_mesh.export(f"{hand_obj_path}/{hand}_obj_{grasp_idx}.obj")
                        np.save(f"{contact_path}/{hand}_contact_pc{grasp_idx}.npy", contact_points_dt)
                        info = [
                                    hand, 
                                    object_id, 
                                    pc_path,
                                    hand_obj_path,
                                    contact_path,
                                    object_path,        
                                    graspit_data['pose'][grasp_idx],                
                                    graspit_data['final_dofs'][grasp_idx],
                                    graspit_data['graspit_dofs'][grasp_idx],
                                    graspit_data['dofs'][grasp_idx],                         
                        ]
                        data_info.append(info)

                # Save DF
                data_info_columns = ['hand', 'object_id', 'pc_path', 'hand_obj_path', 'contact_path', 'object_path', 'hand_pose', 'final_dofs', 'graspit_dofs', 'dofs']
                data_info_df = pd.DataFrame(data_info, columns=data_info_columns)
                data_info_df.to_parquet(f"{data_output_path}/{hand}_metadata.parquet", index=False)


def pose_to_matrix(pose):
    px, py, pz, qw, qx, qy, qz = pose
    rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4)
    T[0:3, 0:3] = rotation
    T[0:3, 3]   = [px, py, pz]
    return T

if __name__ == '__main__':
    main()
