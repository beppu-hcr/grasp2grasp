import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from dataset.data_utils import process_metadata_file, sample_fixed_points
import pickle
from dataset.data_utils import extract_posetheta_from_metadata, get_hand_param

class PairedGraspDataset(Dataset):
    def __init__(self, root_dir, hand1, hand2, split="train", contact=False, gwh=False, jac=False,
                 batch_size=256):
        """
        Args:
            root_dir (str): Root directory of the grasp dataset (e.g., "grasp_data/").
            split (str): Which split to use: "train", "val", or "test".
            split_file (str, optional): Path to the split file. If None, defaults to <root_dir>/split.json.
            train_ratio (float): Fraction of samples to use for training.
            val_ratio (float): Fraction of samples to use for validation.
                The remaining samples will be used for testing.
            transform (callable, optional): Optional transform to be applied on a sample.
            seed (int): Random seed for reproducibility when creating a split.
        """
        self.root_dir = root_dir
        self.hand1 = hand1
        self.hand2 = hand2
        self.split = split
        self.return_contact = contact
        self.batch_size = batch_size
        self.return_gwh = gwh
        self.return_jac = jac

        # Use default split file path if not provided.
        self.split_file = os.path.join(root_dir, hand1, "split.json")
        
        # Collect all object directories.
        hand1_path = os.path.join(root_dir, hand1)
        hand2_path = os.path.join(root_dir, hand2)

        hand1_objects = set(os.listdir(hand1_path))
        hand2_objects = set(os.listdir(hand2_path))
        common_objects = sorted(list(hand1_objects.intersection(hand2_objects)))

        self.object_list = []
        for obj in common_objects:
            path1 = os.path.join(hand1_path, obj)
            path2 = os.path.join(hand2_path, obj)
            if os.path.isdir(path1) and os.path.isdir(path2):
                self.object_list.append((obj, path1, path2))
        
        print(f"Found {len(self.object_list)} common objects between {hand1} and {hand2}.")
        
        # Now split the dataset if needed.
        self._create_or_load_object_split(0.4, 0.1, 0.1)
        
    def _create_or_load_object_split(self, train_ratio, val_ratio, test_ratio):
        # List of all object directories.
        object_keys = [obj[1] for obj in self.object_list]
        # Objects whose base name starts with "0" are forced into training.
        forced_train = [obj for obj in object_keys if os.path.basename(obj).startswith("0")]
        free_objects = [obj for obj in object_keys if not os.path.basename(obj).startswith("0")]
        num_free = len(free_objects)
        
        if os.path.exists(self.split_file):
            with open(self.split_file, 'r') as f:
                split_dict = json.load(f)
            print(f"Loaded split file from {self.split_file}")
            updated = False
            # Ensure forced training objects are in the training split.
            for obj in forced_train:
                if obj not in split_dict.get("train", []):
                    # Remove from any other split if present.
                    if obj in split_dict.get("val", []):
                        split_dict["val"].remove(obj)
                    if obj in split_dict.get("test", []):
                        split_dict["test"].remove(obj)
                    split_dict.setdefault("train", []).append(obj)
                    updated = True
            if updated:
                # Optionally, you can update the file with the changes.
                with open(self.split_file, 'w') as f:
                    json.dump(split_dict, f, indent=4)
                print("Updated split file to ensure forced training objects are in 'train'.")
        else:
            # Shuffle free objects.
            np.random.shuffle(free_objects)
            # Compute counts for free objects.
            train_free_count = int(num_free * train_ratio)
            val_free_count = int(num_free * val_ratio)
            test_free_count = int(num_free * test_ratio)
            # The remaining free objects go to test.
            split_dict = {
                "train": forced_train + free_objects[:train_free_count],
                "val": free_objects[train_free_count:train_free_count+val_free_count],
                "test": free_objects[train_free_count+val_free_count:train_free_count+val_free_count+test_free_count],
                "reserved": free_objects[train_free_count+val_free_count+test_free_count:]
            }
            with open(self.split_file, 'w') as f:
                json.dump(split_dict, f, indent=4)
            print(f"Created new split file at {self.split_file} with forced training objects.")

        # Filter the samples: keep only those from objects in the desired split.
        selected_objects = set(split_dict.get(self.split, []))
        self.samples = []
        for obj, path1, path2 in self.object_list:
            if path1 in selected_objects:
                self.samples.append((obj, path1, path2))
        
        print(f"Using {len(self.samples)} objects from {len(self.object_list)} objects for the '{self.split}' split.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obj_id, path1, path2 = self.samples[idx]
        hand1_latent_all = torch.load(os.path.join(path1, "hand_pc_features.pt")).float()
        hand2_latent_all = torch.load(os.path.join(path2, "hand_pc_features.pt")).float()

        hand1_pose_all, hand1_theta_all = extract_posetheta_from_metadata(path1)
        hand2_pose_all, hand2_theta_all = extract_posetheta_from_metadata(path2)

        contact_all_1 = np.load(os.path.join(path1, "contact_pc/contact_pc_all.npy"), mmap_mode='r').copy()
        success_all_1 = np.load(os.path.join(path1, "contact_pc/success.npy"), mmap_mode='r').copy()
        if self.return_gwh:
            gwh_all_1 = np.load(os.path.join(path1, "gwh.npy"), allow_pickle=True).copy()
            gwh_success_1 = np.load(os.path.join(path1, "hulls_success.npy"), mmap_mode='r').copy()
            success_all_1 = np.logical_and(success_all_1, gwh_success_1).astype(np.float32)
        if self.return_jac:
            jac_all_1 = np.load(os.path.join(path1, "jacobian.npy"), mmap_mode='r').copy()
            jac_success_1 = np.load(os.path.join(path1, "jacobian_success.npy"), mmap_mode='r').copy()
            success_all_1 = np.logical_and(success_all_1, jac_success_1).astype(np.float32)

        contact_all_2 = np.load(os.path.join(path2, "contact_pc/contact_pc_all.npy"), mmap_mode='r').copy()
        success_all_2 = np.load(os.path.join(path2, "contact_pc/success.npy"), mmap_mode='r').copy()
        if self.return_gwh:
            gwh_all_2 = np.load(os.path.join(path2, "gwh.npy"), allow_pickle=True).copy()
            gwh_success_2 = np.load(os.path.join(path2, "hulls_success.npy"), mmap_mode='r').copy()
            success_all_2 = np.logical_and(success_all_2, gwh_success_2).astype(np.float32)
        if self.return_jac:
            jac_all_2 = np.load(os.path.join(path2, "jacobian.npy"), mmap_mode='r').copy()
            jac_success_2 = np.load(os.path.join(path2, "jacobian_success.npy"), mmap_mode='r').copy()
            success_all_2 = np.logical_and(success_all_2, jac_success_2).astype(np.float32)

        n1, n2 = contact_all_1.shape[0], contact_all_2.shape[0]
        assert n1 == hand1_latent_all.size(0)
        assert n2 == hand2_latent_all.size(0)
        if self.return_gwh:
            assert n1 == gwh_all_1.shape[0]
            assert n2 == gwh_all_2.shape[0]
        if self.return_jac:
            assert n1 == jac_all_1.shape[0]
            assert n2 == jac_all_2.shape[0]
        indices1 = np.random.choice(n1, self.batch_size, replace=False, p=success_all_1/np.sum(success_all_1)) 
        indices2 = np.random.choice(n2, self.batch_size, replace=False, p=success_all_2/np.sum(success_all_2))

        if self.return_contact:
            contact_pc_all_1 = contact_all_1[indices1]
            contact_pc_all_2 = contact_all_2[indices2]
        if self.return_gwh:
            gwh_1 = gwh_all_1[indices1]
            gwh_2 = gwh_all_2[indices2]
        if self.return_jac:
            jac_1 = jac_all_1[indices1]
            jac_2 = jac_all_2[indices2]

        hand1_latent = hand1_latent_all[indices1]
        hand2_latent = hand2_latent_all[indices2]

        object_latent_idx = np.random.randint(0, 16)
        with open(os.path.join(self.root_dir, f"object_pc/{obj_id}/pc_norm_latent_LION_{object_latent_idx:03d}.pk"), 'rb') as f:
            object_latent = pickle.load(f)
        
        object_global_latent = torch.tensor(object_latent[0][1]).float().repeat(self.batch_size, 1)
        object_local_latent = torch.tensor(object_latent[1][1]).float().repeat(self.batch_size, 1)

        hand1_pose = torch.tensor(hand1_pose_all[indices1])
        hand1_theta = torch.tensor(hand1_theta_all[indices1])
        hand2_pose = torch.tensor(hand2_pose_all[indices2])
        hand2_theta = torch.tensor(hand2_theta_all[indices2])

        hand1_param = get_hand_param(hand1_pose, hand1_theta)
        hand2_param = get_hand_param(hand2_pose, hand2_theta)

        sample = {
            'indices1': torch.tensor(indices1),
            'indices2': torch.tensor(indices2),
            'hand1_latent': hand1_latent,
            'hand2_latent': hand2_latent,
            'object_global_latent': object_global_latent,
            'object_local_latent': object_local_latent,
            'hand1_param': hand1_param,
            'hand2_param': hand2_param,
        }
        if self.return_contact:
            sample['contact_pc_1'] = torch.tensor(contact_pc_all_1).float()
            sample['contact_pc_2'] = torch.tensor(contact_pc_all_2).float()
        if self.return_gwh:
            sample['gwh_1'] = gwh_1
            sample['gwh_2'] = gwh_2
        if self.return_jac:
            sample['jac_1'] = torch.tensor(jac_1).float()
            sample['jac_2'] = torch.tensor(jac_2).float()

        return sample
    
if __name__ == "__main__":
    dataset = PairedGraspDataset(root_dir='/data/XXX/data/grasp_data', hand1="HumanHand", hand2="Allegro", split="train", contact=True, batch_size=256)
    for epoch in range(10):
        print(f"Epoch {epoch}")
        for i in tqdm(range(len(dataset))):
            sample = dataset[i]
            if i == 0:
                print(sample['indices1'][:5])
                print(sample["hand1_latent"].shape)
                print(sample["hand2_latent"].shape)
                print(sample["object_global_latent"].shape)
                print(sample["object_local_latent"].shape)
                print(sample["hand1_param"].shape)
                print(sample["hand2_param"].shape)
                if "contact_pc_1" in sample:
                    print(sample["contact_pc_1"].shape)
                    print(sample["contact_pc_2"].shape)
            
