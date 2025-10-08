from copy import deepcopy
from glob import glob
import json
import multiprocessing as mp
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.io as sio
from scipy.spatial.transform import Rotation as R
import shutil
import trimesh
from urdfpy import URDF
# from read_error_list import extract_error_objects

from contact_utils import *


"""
TODO
2. Create dict of object id to object pc
3. Update this to include object pc path
"""

NUM_WORKERS = mp.cpu_count()


def main():
    hands = {
                # "shadow_hand": ["shadow_hand"]
                # "Allegro": ["allegro_hand_description_right"], 
                "HumanHand": ["HumanHand"]
    }
    # urdf_base_path = "/workspace/code/Point-VAE/isaac_sim_grasping/grippers"
    urdf_base_path = Path(__file__).parent.parent.parent / "grippers"
    # graspit_base_path = "/diskstation/XXX/data/multigripper_grasp_data/Dataset/graspit_grasps"
    graspit_base_path = Path(__file__).parent.parent.parent / "data/multigripper_grasp_data/Dataset/graspit_grasps"
    # output_path = "/diskstation/XXX/data/grasp_data" 
    output_path = Path(__file__).parent.parent.parent / "data/grasp_data"
    contact_threshold = 0.01

    # error_path = "/diskstation/XXX/data/grasp_data/HumanHand/error_log.txt"  # ensure this is the correct path to your error log file
    # error_objects = extract_error_objects(error_path)

    for hand, hand_urdf_names in hands.items():
        Path.mkdir(Path(f"{output_path}/{hand}"), parents=True, exist_ok=True)

        for hand_urdf_name in hand_urdf_names:
            print(f"Processing {hand_urdf_name}")
            data_output_path = f"{output_path}/{hand}"
            Path.mkdir(Path(data_output_path), exist_ok=True)

            urdf_path = f"{urdf_base_path}/{hand}"
            robot = URDF.load(f'{urdf_path}/{hand_urdf_name}.urdf')
            # non_stationary_joints = [joint for joint in robot.joints if 'tip' not in joint.name]
            # non_stationary_joints = ['little_finger_joint1', 'little_finger_joint2', 'little_finger_joint3', 'little_finger_joint4', 'little_finger_joint5',
            #        'ring_finger_joint1', 'ring_finger_joint2', 'ring_finger_joint3', 'ring_finger_joint4', 
            #        'middle_finger_joint1', 'middle_finger_joint2', 'middle_finger_joint3', 'middle_finger_joint4',
            #        'index_finger_joint1', 'index_finger_join2', 'index_finger_joint3', 'index_finger_joint4',
            #        'thumb_joint1', 'thumb_joint2', 'thumb_joint3', 'thumb_joint4', 'thumb_joint5']
            non_stationary_joints = ['palm_index1_0_joint', 'index1_0_index1_joint', 'index1_index2_joint', 'index2_index3_joint',
                       'palm_mid1_0_joint', 'mid1_0_mid1_joint', 'mid1_mid2_joint', 'mid2_mid3_joint',
                       'palm_ring1_0_joint', 'ring1_0_ring1_joint', 'ring1_ring2_joint', 'ring2_ring3_joint',
                       'palm_pinky1_0_joint', 'pinky1_0_pinky1_joint', 'pinky1_pinky2_joint', 'pinky2_pinky3_joint',
                       'palm_thumb1_0_joint', 'thumb1_0_thumb1_joint', 'thumb1_thumb2_joint', 'thumb2_thumb3_joint']

            shared_dict = {
                'hand_urdf_name': hand_urdf_name,
                'urdf_path': urdf_path,
                'robot': robot,
                'non_stationary_joints': non_stationary_joints,
                'contact_threshold': contact_threshold,
                'data_output_path': data_output_path
                # 'error_objects': error_objects
            }

            graspit_paths = glob(f'{graspit_base_path}/{hand}/*.json')
            args_list = [(graspit_path, shared_dict) for graspit_path in graspit_paths]
            with mp.Pool(processes=NUM_WORKERS) as pool:
                pool.starmap(process_graspit_data, args_list)

        print(f'{hand} Data Generation Complete')
    print('All Data Generation Complete!')


def pose_to_matrix(pose):
    px, py, pz, qw, qx, qy, qz = pose
    rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4)
    T[0:3, 0:3] = rotation
    T[0:3, 3]   = [px, py, pz]
    return T

def process_graspit_data(graspit_path, shared_dict):
    # Extract vars
    hand_name = shared_dict['hand_urdf_name']
    urdf_path = shared_dict['urdf_path']
    robot = shared_dict['robot']
    non_stationary_joints = shared_dict['non_stationary_joints']
    contact_threshold = shared_dict['contact_threshold']
    data_output_path = shared_dict['data_output_path']    
    data_info = []

    try:
        with open(graspit_path, 'r') as file:
            graspit_data = json.load(file)

        object_id = graspit_data['object_id']
        # if object_id not in shared_dict['error_objects']:
        #     print(f"\tPass: {object_id}")
        #     return
        print(f"\tProcessing: {object_id}")

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
        base_object_path = Path(__file__).parent.parent.parent / "data/mgg_pc/objects/npy"
        src_object_path = f"{base_object_path}/{object_id}.npy"
        object_path = f"{output_path}/{object_id}.npy"
        # shutil.copy(src_object_path, object_path)

        # Read Object Point Cloud (XYZ)
        object_points = np.load(src_object_path)
        kd_tree, obj_pcd = build_kdtree_for_object(object_points) # shape: (N,3)
        np.save(object_path, sample_fixed_points(object_points, 2048))
        
        for grasp_idx in range(len(graspit_data['pose'])):
            # Hand Pose
            hand_pose = graspit_data['pose'][grasp_idx]
            hand_T = pose_to_matrix(hand_pose)

            # Link poses
            joint_mapping = {} 
            for i, joint in enumerate(non_stationary_joints):
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

                        scale = visual.geometry.mesh.scale
                        mesh.apply_scale(scale)

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
                                                                    contact_threshold)
                if contact_points_dt.shape[0] == 0:
                    continue
                contact_points_dt = sample_fixed_points(contact_points_dt, 256)

                # Hand Class flag
                ones_col = np.zeros((points.shape[0], 1))
                points = np.hstack([points, ones_col])

                np.save(f"{pc_path}/{hand_name}_pc_{grasp_idx}.npy", points)
                # hand_mesh.export(f"{hand_obj_path}/{hand_name}_obj_{grasp_idx}.obj")
                np.save(f"{contact_path}/{hand_name}_contact_pc{grasp_idx}.npy", contact_points_dt)
                info = [
                            hand_name, 
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
        data_info_df.to_parquet(f"{output_path}/metadata.parquet", index=False)
    except Exception as e:
        with open(f"{data_output_path}/error_log.txt", "a") as log_file:
            log_file.write(f'Error Processing: {hand_name} - {object_id}\n\tException: {e}\n')

        if len(data_info) > 0:
            data_info_columns = ['hand', 'object_id', 'pc_path', 'hand_obj_path', 'contact_path', 'object_path', 'hand_pose', 'final_dofs', 'graspit_dofs', 'dofs']
            data_info_df = pd.DataFrame(data_info, columns=data_info_columns)
            data_info_df.to_parquet(f"{output_path}/metadata.parquet", index=False)
            

if __name__ == '__main__':
    main()
