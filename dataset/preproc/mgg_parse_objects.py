import numpy as np
from pathlib import Path
import scipy.io as sio
from tqdm import tqdm
import trimesh
from urdfpy import URDF
from lxml import etree


def main():
    object_models_path = Path(__file__).parent.parent.parent / 'data/multigripper_grasp_data/Dataset/Object_Models'
    google_object_path = f'{object_models_path}/GoogleScannedObjects'
    ycb_object_path = f'{object_models_path}/YCB'
    output_path = Path(__file__).parent.parent.parent / 'data/mgg_pc/objects'
    Path.mkdir(Path(output_path), exist_ok=True)
    Path.mkdir(Path(f'{output_path}/mat'), exist_ok=True)
    Path.mkdir(Path(f'{output_path}/npy'), exist_ok=True)
    Path.mkdir(Path(f'{output_path}/obj'), exist_ok=True)

    model_ids_path = f'{object_models_path}/mgg_models_ids.txt'
    google_ids_path = f'{object_models_path}/GoogleScannedObjects_model_ids.txt'
    ycb_ids_path = f'{object_models_path}/ycb_object_ids.txt'
    model_ids = get_ids(model_ids_path)
    google_ids = get_ids(google_ids_path)
    ycb_ids = get_ids(ycb_ids_path)

    pbar = tqdm(model_ids)
    for model_id in pbar:
        pbar.set_description(f'Processing: {model_id}')
        if model_id in google_ids:
            urdf_path = f'{google_object_path}/{model_id}'
            urdf_name = f'{model_id}.urdf'
            object_pc, object_mesh = load_urdf(urdf_path, urdf_name)
        elif model_id in ycb_ids:
            object_path = f'{ycb_object_path}/{model_id}'
            object_pc = np.loadtxt(f'{object_path}/points.xyz', delimiter=' ')
            object_mesh = trimesh.load(f'{object_path}/textured.obj')
        else:
            raise Exception(f'Model ID: {model_id} Not Found!')
        
        # Object Class flag
        ones_col = np.zeros((object_pc.shape[0], 0))
        object_pc = np.hstack([object_pc, ones_col])
        
        # Save Data
        np.save(f"{output_path}/npy/{model_id}.npy", object_pc)
        sio.savemat(f"{output_path}/mat/{model_id}.mat", {"points": object_pc})
        object_mesh.export(f"{output_path}/obj/{model_id}.obj")

def get_ids(file_path):
    with open(file_path, 'r') as f:
        ids = [line.strip() for line in f]
    return set(ids)

def load_urdf(urdf_path, urdf_name):    
    # The GoogleScannedObjects URDFs comment out <mass>, which urdfpy requires in <inertial>.
    # Only the visual meshes are used here, so drop inertial blocks without a mass.
    parser = etree.XMLParser(remove_comments=True, remove_blank_text=True)
    root = etree.parse(f'{urdf_path}/{urdf_name}', parser=parser).getroot()
    for inertial in list(root.iter('inertial')):
        if inertial.find('mass') is None:
            inertial.getparent().remove(inertial)
    robot = URDF._from_xml(root, urdf_path)

    scene = []
    link_poses = robot.link_fk()
    for link in robot.links:
        link_pose_global = link_poses[link]
        for visual in link.visuals:
            if visual.geometry.mesh is not None:
                mesh_file = visual.geometry.mesh.filename
                mesh_file = f'{urdf_path}/{mesh_file}'
                mesh = trimesh.load(mesh_file)

                local_tf = visual.origin  
                mesh_tf = link_pose_global @ local_tf            

                # Apply the local transform
                mesh.apply_transform(mesh_tf)
                scene.append(mesh)

    # Sample Mesh Points
    object_mesh = trimesh.util.concatenate(scene)
    points, _ = trimesh.sample.sample_surface(object_mesh, 4096)
    points = np.vstack(points)

    # Append Class flag
    if points.shape[0] == 0:
        raise Exception(f'Unable to sample pc for {urdf_path}')
    
    return points, object_mesh


if __name__ == '__main__':
    main()
