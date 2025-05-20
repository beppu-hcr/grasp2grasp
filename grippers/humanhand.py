import numpy as np
import torch
import os
import trimesh
import glob
from functools import cached_property

import pytorch_kinematics as pk
from grippers.utils import quaternion_to_matrix
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d

class HumanHandLayer(torch.nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HumanHand/HumanHand.urdf")
        self.learnable_robot_model = pk.build_chain_from_urdf(open(urdf_path).read().encode('ascii')).to(device=device)
        self.device = device
        self.meshes = self.load_meshes()
        self.link_names = ["palm", "index1", "index2", "index3",
                           "mid1", "mid2", "mid3", 
                           "ring1", "ring2", "ring3",
                           "pinky1", "pinky2", "pinky3",
                           "thumb1", "thumb2", "thumb3"]
        self.name2name = {"palm":"palm",
            "index1":"index1", "index2":"index2", "index3":"index3",
            "mid1":"mid1", "mid2":"mid2", "mid3":"mid3",
            "ring1":"ring1", "ring2":"ring2", "ring3":"ring3",
            "pinky1":"pinky1", "pinky2":"pinky2", "pinky3":"pinky3",
            "thumb1":"thumb1", "thumb2":"thumb2", "thumb3":"thumb3"}
        
        # self.joint_names = ['little_finger_joint1', 'little_finger_joint2', 'little_finger_joint3', 'little_finger_joint4', 'little_finger_joint5',
        #            'ring_finger_joint1', 'ring_finger_joint2', 'ring_finger_joint3', 'ring_finger_joint4', 
        #            'middle_finger_joint1', 'middle_finger_joint2', 'middle_finger_joint3', 'middle_finger_joint4',
        #            'index_finger_joint1', 'index_finger_join2', 'index_finger_joint3', 'index_finger_joint4',
        #            'thumb_joint1', 'thumb_joint2', 'thumb_joint3', 'thumb_joint4', 'thumb_joint5']
        self.joint_names = ['palm_index1_0_joint', 'index1_0_index1_joint', 'index1_index2_joint', 'index2_index3_joint',
                       'palm_mid1_0_joint', 'mid1_0_mid1_joint', 'mid1_mid2_joint', 'mid2_mid3_joint',
                       'palm_ring1_0_joint', 'ring1_0_ring1_joint', 'ring1_ring2_joint', 'ring2_ring3_joint',
                       'palm_pinky1_0_joint', 'pinky1_0_pinky1_joint', 'pinky1_pinky2_joint', 'pinky2_pinky3_joint',
                       'palm_thumb1_0_joint', 'thumb1_0_thumb1_joint', 'thumb1_thumb2_joint', 'thumb2_thumb3_joint']

    def load_meshes(self):
        mesh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"HumanHand/meshes/*.stl")
        # print(mesh_path)
        mesh_files = glob.glob(mesh_path)
        mesh_files = [f for f in mesh_files if os.path.isfile(f)]
        meshes = {}
        for mesh_file in mesh_files:
            name = os.path.basename(mesh_file)[:-4]
            mesh = trimesh.load(mesh_file)
            mesh.apply_scale(0.001)
            # meshes[name] = torch.FloatTensor(mesh.vertices).to(self.device)
            vertices = torch.FloatTensor(mesh.vertices).to(self.device)
            faces = torch.LongTensor(mesh.faces).to(self.device)
            meshes[name] = (vertices, faces)
        return meshes
    
    @cached_property
    def concatenated_faces(self):
        """
        Returns a tensor containing the concatenated faces of all link meshes.
        For each link, face indices are offset by the cumulative number of vertices from previous links.
        """
        offset = 0
        faces_list = []
        for name in self.link_names:
            # Retrieve the mesh using the name mapping
            vertices, faces = self.meshes[self.name2name[name]]
            faces_list.append(faces + offset)
            offset += vertices.shape[0]
        return torch.cat(faces_list, dim=0)
    
    def get_hand_param(self, pose, theta):
        batch_size = pose.shape[0]
        if pose.shape[1] == 9:
            return torch.cat([pose, theta], dim=1)
        elif pose.shape[1] == 7:
            rot_matrix = quaternion_to_matrix(pose[:, 3:7])
            rot_6d = matrix_to_rotation_6d(rot_matrix)
            return torch.cat([pose[:, :3], rot_6d, theta], dim=1)
        else:
            raise ValueError("Pose should be of shape (batch_size x 7) or (batch_size x 9)")

    def forward(self, pose, theta):
        """[summary]
        Args:
            pose (Tensor (batch_size x 7)): The pose of the base link of the hand as a translation matrix.
            theta (Tensor (batch_size x 20)): The 20 degrees of freedome of the Barrett hand. The first column specifies the angle between 
            fingers F1 and F2,  the second to fourth column specifies the joint angle around the proximal link of each finger while the fifth
            to the last column specifies the joint angle around the distal link for each finger

       """
        batch_size = pose.shape[0]
        pose_matrix = torch.zeros((batch_size, 4, 4), device=self.device)
        pose_matrix[:, 3, 3] = 1
        if pose.shape[1] == 9:
            pose_matrix[:, :3, :3] = rotation_6d_to_matrix(pose[:, 3:9])
        elif pose.shape[1] == 7:
            pose_matrix[:, :3, :3] = quaternion_to_matrix(pose[:, 3:7])
        else:
            raise ValueError("Pose should be of shape (batch_size x 7) or (batch_size x 9)")
        pose_matrix[:, :3, 3] = pose[:, :3]

        theta_dict = {jname:theta[:, i] for i, jname in enumerate(self.joint_names)}
        fk = self.learnable_robot_model.forward_kinematics(theta_dict)
        #print(f"Time compute fk: {time.time() - curr}")
        #curr = time.time()

        all_verts = []

        for name in self.link_names:
            link_vertices = self.meshes[self.name2name[name]][0].repeat(batch_size,1,1)
            link_vertices = fk[name].transform_points(link_vertices)
            link_vertices = torch.matmul(pose_matrix[:, :3, :3], link_vertices.transpose(1,2)).transpose(1,2) + pose_matrix[:, :3, 3].unsqueeze(1)
            all_verts.append(link_vertices)
        
        all_verts = torch.cat(all_verts, dim=1)
        return all_verts

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hand = HumanHandLayer(device=device)
    pose = torch.tensor([[0.10671630136585901, 0.14722218055396602, 0.025545334709028002, 0.930735949279435, -0.12190149845504201, -0.22712721989595103, 0.25939129394731003]]).to(device)
    theta = torch.tensor([[0.349066, 0.23296699999999998, 0.23296699999999998, 0.23296699999999998, -1.38457e-16, 0.416717, 0.416717, 0.416717, -0.349066, 0.814217, 0.814217, 0.814217, -0.5235989999999999, 1.35172, 1.35172, 1.35172, -0.174533, -0.682994, 0.58875, 0.58875]]).to(device)
    # pose = torch.randn(64, 7).to(device)
    # theta = torch.randn(64, 20).to(device)
    # print(pose, theta)
    verts = hand(pose, theta)
    print("Vertices shape:", verts.shape)
    print("Concatenated faces shape:", hand.concatenated_faces.shape)
    verts_np = verts.cpu().numpy()
    hand_mesh = trimesh.Trimesh(vertices=verts_np[0], faces=hand.concatenated_faces.cpu().numpy())
    # hand_mesh.export("test_output/dataloader/humanhand_2.obj")
    # np.savetxt("shadow_hand.xyz", verts_np[0], delimiter=" ")
