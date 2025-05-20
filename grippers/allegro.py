import numpy as np
import torch
import os
import trimesh
import glob
from functools import cached_property

import pytorch_kinematics as pk
from grippers.utils import quaternion_to_matrix
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d

class AllegroLayer(torch.nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Allegro/allegro_hand_description_right.urdf")
        self.learnable_robot_model = pk.build_chain_from_urdf(open(urdf_path).read().encode('ascii')).to(device=device)
        self.device = device
        self.meshes = self.load_meshes()
        self.link_names = ["base_link"] + [f"link_{i}.0" for i in range(16)] + ["link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip"]
        self.name2name = {"base_link":"base_link",
            "link_0.0":"link_0.0", "link_1.0":"link_1.0", "link_2.0":"link_2.0", "link_3.0":"link_3.0", "link_3.0_tip":"link_3.0_tip",
            "link_4.0":"link_0.0", "link_5.0":"link_1.0", "link_6.0":"link_2.0", "link_7.0":"link_3.0", "link_7.0_tip":"link_3.0_tip",
            "link_8.0":"link_0.0", "link_9.0":"link_1.0", "link_10.0":"link_2.0", "link_11.0":"link_3.0", "link_11.0_tip":"link_3.0_tip",
            "link_12.0":"link_12.0_right", "link_13.0":"link_13.0", "link_14.0":"link_14.0", "link_15.0":"link_15.0", "link_15.0_tip":"link_15.0_tip"}
        self.joint_names = self.learnable_robot_model.get_joint_parameter_names()

    def load_meshes(self):
        mesh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"Allegro/meshes_simplified/*.STL")
        # print(mesh_path)
        mesh_files = glob.glob(mesh_path)
        mesh_files = [f for f in mesh_files if os.path.isfile(f)]
        meshes = {}
        for mesh_file in mesh_files:
            if "_left" in mesh_file:
                continue
            name = os.path.basename(mesh_file)[:-4]
            mesh = trimesh.load(mesh_file)
            #mesh.apply_scale(0.001)
            # meshes[name] = torch.FloatTensor(mesh.vertices).to(self.device),
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
            theta (Tensor (batch_size x 16)): The 16 degrees of freedome of the Barrett hand. The first column specifies the angle between 
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
    hand = AllegroLayer(device=device)
    pose = torch.tensor([0.017953825588269003, -0.149046312384636, 0.038230525091728, -0.32584274118711104, -0.27448123282775, -0.880297485099072, -0.20871726945625402]).unsqueeze(0).to(device)

    # theta = torch.tensor([0.235781, 0.52978, 0.551782, 0.498782, 0.129141, 0.42314, 0.44514099999999995, 0.39214099999999996, 0.380156, 0.674155, 0.6961569999999999, 0.643157, 0.8163600000000001, 0.24836000000000003, 0.164359, 0.191359]).unsqueeze(0).to(device)
    theta = torch.tensor([0.105061,    0.492805,    0.525522,    0.486586,    0.0142796,
    0.402024  ,  0.434741  ,  0.395804   , 0.57    ,    1.54937,     1.58208,
    1.54315  ,   0.549315  ,  0.299201  ,  0.406547 ,   0.434127  ]).unsqueeze(0).to(device)
    # pose = torch.zeros(1, 9).to(device)
    # pose[:, 3] = 1.0
    # pose[:, 7] = 1.0
    # theta = torch.zeros(1, 16).to(device)
    # pose[0, -1] = 1.0
#     # pose = torch.randn(64, 7).to(device)
    # theta = torch.randn(64, 16).to(device)
    # print(pose, theta)
    verts = hand(pose, theta)
    print("Vertices shape:", verts.shape)
    print("Concatenated faces shape:", hand.concatenated_faces.shape)
    verts_np = verts.cpu().numpy()
    hand_mesh = trimesh.Trimesh(vertices=verts_np[0], faces=hand.concatenated_faces.cpu().numpy())
    hand_mesh.export("test_output/gdg_debug.obj")
    # np.savetxt("test_output/hand_3.xyz", verts_np[0], delimiter=" ")
