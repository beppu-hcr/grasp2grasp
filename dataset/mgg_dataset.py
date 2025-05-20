import os
import json
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataset.data_utils import process_metadata_file, sample_fixed_points

class GraspDataset(Dataset):
    def __init__(self, root_dir, hand, split="train", seed=0, contact=False, val_subsample_ratio=0.1):
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
        self.split = split
        self.seed = seed
        self.return_contact = contact
        self.val_subsample_ratio = val_subsample_ratio

        # Use default split file path if not provided.
        self.split_file = os.path.join(root_dir, hand, "split.json")
        
        # # Load metadata from all samples.
        # self.object_samples = {}
        # # Sort the hand and object directories to ensure consistent ordering.
        # hand_path = os.path.join(root_dir, hand)
        # if os.path.isdir(hand_path):
        #     for obj_dir in tqdm(sorted(os.listdir(hand_path))):
        #         obj_path = os.path.join(hand_path, obj_dir)
        #         if os.path.isdir(obj_path):
        #             meta_file = os.path.join(obj_path, 'metadata.parquet')
        #             if os.path.exists(meta_file):
        #                 df = pd.read_parquet(meta_file)
        #                 samples = []
        #                 for grasp_idx, row in df.iterrows():
        #                     sample_info = row.to_dict()
        #                     sample_info['base_dir'] = obj_path
        #                     sample_info['grasp_idx'] = grasp_idx
        #                     samples.append(sample_info)
        #                 if samples:
        #                     self.object_samples[obj_path] = samples
        
        # print(f"Loaded metadata for {len(self.object_samples)} objects.")
        # Collect all object directories.
        object_dirs = []
        hand_path = os.path.join(root_dir, hand)
        if os.path.isdir(hand_path):
            for obj_dir in sorted(os.listdir(hand_path)):
                obj_path = os.path.join(hand_path, obj_dir)
                if os.path.isdir(obj_path):
                    object_dirs.append(obj_path)
                        
        print(f"Found {len(object_dirs)} object directories. Loading metadata...")

        # Use a thread pool to load metadata in parallel.
        self.object_samples = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_metadata_file, obj_path): obj_path for obj_path in object_dirs}
            for future in tqdm(as_completed(futures), total=len(futures)):
                obj_path, samples = future.result()
                if samples is not None and samples:
                    self.object_samples[obj_path] = samples

        print(f"Loaded metadata for {len(self.object_samples)} objects.")

        # Preload object point clouds into memory.
        self.preloaded_object_point_clouds = {}
        for obj_path in tqdm(self.object_samples.keys(), desc="Preloading object point clouds"):
            obj_pc_path = os.path.join(obj_path, f'{os.path.basename(obj_path)}.npy')
            if os.path.exists(obj_pc_path):
                # Load once and copy to ensure writability.
                self.preloaded_object_point_clouds[obj_path] = sample_fixed_points(np.load(obj_pc_path).copy(), 2048)
            else:
                print(f"Warning: Object point cloud file not found at {obj_pc_path}")
        
        # Now split the dataset if needed.
        self._create_or_load_object_split(0.4, 0.1, 0.1)
        
    def _create_or_load_object_split(self, train_ratio, val_ratio, test_ratio):
        # List of all object directories.
        object_keys = list(self.object_samples.keys())
        # Objects whose base name starts with "0" are forced into training.
        forced_train = [obj for obj in object_keys if os.path.basename(obj).startswith("0")]
        free_objects = [obj for obj in object_keys if not os.path.basename(obj).startswith("0")]
        num_free = len(free_objects)
        
        if os.path.exists(self.split_file):
            with open(self.split_file, 'r') as f:
                split_dict = json.load(f)
            print(f"Loaded split file from {self.split_file}")
            # Ensure forced training objects are in the training split.
            for obj in forced_train:
                if obj not in split_dict.get("train", []):
                    # Remove from any other split if present.
                    if obj in split_dict.get("val", []):
                        split_dict["val"].remove(obj)
                    if obj in split_dict.get("test", []):
                        split_dict["test"].remove(obj)
                    split_dict.setdefault("train", []).append(obj)
            # Optionally, you can update the file with the changes.
            with open(self.split_file, 'w') as f:
                json.dump(split_dict, f, indent=4)
            print("Updated split file to ensure forced training objects are in 'train'.")
        else:
            # Shuffle free objects.
            np.random.seed(self.seed)
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
        for obj_path, samples in self.object_samples.items():
            if obj_path in selected_objects:
                self.samples.extend(samples)
        
        # Sub sample if val or test to reduce eval time
        if (self.split == "val" or self.split == "test") and 0 < self.val_subsample_ratio < 1:
            selected_idx = np.random.choice(len(self.samples), int(len(self.samples) * self.val_subsample_ratio), replace=False)
            self.samples = [self.samples[i] for i in selected_idx]
        print(f"Using {len(self.samples)} samples from {len(selected_objects)} objects for the '{self.split}' split.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_meta = self.samples[idx]
        base_dir = sample_meta['base_dir']
        hand_name = sample_meta['hand']
        grasp_idx = sample_meta['grasp_idx']

        # Load object point cloud (assumed to be stored as 'obj_pc.npy' in the object directory)
        object_pc = self.preloaded_object_point_clouds[base_dir]

        # Load hand point cloud using the file name from metadata (assumed to be in the 'hand_pc' folder)
        hand_pc_path = os.path.join(base_dir, "hand_pc", f"{hand_name}_pc_{grasp_idx}.npy")
        if not os.path.exists(hand_pc_path):
            hand_pc_path = os.path.join(base_dir, "hand_pc", f"{hand_name}_pc__{grasp_idx}.npy")
        hand_pc = np.load(hand_pc_path, mmap_mode='r').copy()
        hand_pc = hand_pc[:, :3]

        # TODO: Fix empty contact point cloud
        if self.return_contact:
            contact_pc_path = os.path.join(base_dir, "contact_pc", f"{hand_name}_contact_pc{grasp_idx}.npy")
            if not os.path.exists(contact_pc_path):
                contact_pc_path = os.path.join(base_dir, "contact_pc", f"{hand_name}_contact_pc_{grasp_idx}.npy")
            contact_pc = np.load(contact_pc_path, mmap_mode='r').copy()
            if contact_pc.shape[0] == 0:
                # print(f"Empty contact point cloud for {hand_name}_contact_pc{grasp_idx}.npy")
                return self.__getitem__(np.random.randint(len(self)))
            contact_pc = sample_fixed_points(contact_pc, 256)

        # Get hand pose and degrees-of-freedom from the metadata.
        hand_pose = sample_meta['hand_pose'].copy()
        hand_dofs = sample_meta['graspit_dofs'].copy()

        sample = {
            'grasp_idx': grasp_idx,
            'object_pc': object_pc,
            'hand_pc': hand_pc,
            'hand_pose': hand_pose,
            'hand_dofs': hand_dofs
        }
        if self.return_contact:
            sample['contact_pc'] = contact_pc

        return sample

# Example usage:
# dataset_train = GraspDataset(root_dir='path/to/grasp_data', split="train")
# dataset_val = GraspDataset(root_dir='path/to/grasp_data', split="val")
# dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True)
if __name__ == "__main__":
    dataset = GraspDataset(root_dir='/data/XXX/data/grasp_data', hand="HumanHand", split="train")
    for idx in [6500, 13000, 19500]:
        # idx = np.random.randint(len(dataset))
        sample = dataset[idx]
        print(sample.keys())
        print(sample['object_pc'].shape)
        print(sample['hand_pc'].shape)
        # print(sample['contact_pc'].shape)
        print(sample['hand_pose'].shape)
        print(sample['hand_dofs'].shape)

        object_pc = sample['object_pc']
        hand_pc = sample['hand_pc']
        hand_pose = sample['hand_pose']
        hand_dofs = sample['hand_dofs']
        print(hand_pose.tolist())
        print(hand_dofs.tolist())
        np.savetxt(f'test_output/dataloader/object_pc_{idx}.xyz', object_pc, delimiter=' ')
        np.savetxt(f'test_output/dataloader/hand_pc_{idx}.xyz', hand_pc, delimiter=' ')

    # from torch.utils.data import DataLoader
    # dataloader = DataLoader(dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=8, persistent_workers=True)
    # for batch in tqdm(dataloader):
    #     object_pc = batch['object_pc'].cuda()
    #     hand_pc = batch['hand_pc'].cuda()
    #     # contact_pc = batch['contact_pc'].cuda()
    #     hand_pose = batch['hand_pose'].cuda()
    #     hand_dofs = batch['hand_dofs'].cuda()
