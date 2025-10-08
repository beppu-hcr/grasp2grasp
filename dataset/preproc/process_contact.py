import os
from os.path import join as pjoin
import numpy as np
import re
from multiprocessing import Pool
import argparse
from tqdm import tqdm

def sample_fixed_points(points, num_points):
    """
    Sample the input point cloud so that it has exactly num_points.
    If points has more than num_points, sample without replacement.
    If fewer, include all original points and sample additional points (with replacement)
    to reach num_points, then shuffle the result.
    """
    N = points.shape[0]
    if N == num_points:
        return points
    elif N > num_points:
        indices = np.random.choice(N, num_points, replace=False)
        return points[indices]
    else:
        # Include all original points
        additional = points[np.random.choice(N, num_points - N, replace=True)]
        combined = np.concatenate([points, additional], axis=0)
        # Shuffle so the duplicated points are not always at the end
        np.random.shuffle(combined)
        return combined

def process_object_dir(func_args):
    """
    Process a single object directory by loading, concatenating, 
    and saving the contact_pc files.
    """
    object_dir, hand = func_args

    print(f"Processing {object_dir} for {hand} hand.")
    contact_folder = os.path.join(object_dir, 'contact_pc')
    if not os.path.isdir(contact_folder):
        print(f"Warning: {contact_folder} is not a directory.")
        return

    # List files matching the pattern "contact_pc_<index>.npy"
    if hand == "Allegro":
        files = [f for f in os.listdir(contact_folder) if re.match(r'^allegro_hand_description_right_contact_pc(\d+)\.npy$', f)]
    elif hand == "HumanHand":
        files = [f for f in os.listdir(contact_folder) if re.match(r'^HumanHand_contact_pc(\d+)\.npy$', f)]
    elif hand == "shadow_hand":
        files = [f for f in os.listdir(contact_folder) if re.match(r'^shadow_hand_contact_pc(\d+)\.npy$', f)]
    else:
        raise ValueError(f"Unknown hand type: {hand}")
    
    if not files:
        print(f"No contact_pc files found in {contact_folder}.")
        return

    # Sort files by numerical index to maintain order
    def extract_index(filename):
        if hand == "Allegro":
            match = re.match(r'^allegro_hand_description_right_contact_pc(\d+)\.npy$', filename)
        elif hand == "HumanHand":
            match = re.match(r'^HumanHand_contact_pc(\d+)\.npy$', filename)
        elif hand == "shadow_hand":
            match = re.match(r'^shadow_hand_contact_pc(\d+)\.npy$', filename)
        return int(match.group(1)) if match else float('inf')

    files.sort(key=extract_index)

    # Load each numpy array
    arrays = []
    success = []
    for file in files:
        file_path = os.path.join(contact_folder, file)
        try:
            arr = np.load(file_path)
            if arr.shape[0] == 0:
                arr = np.zeros((256, 3))
                success.append(0)
            else:
                arr = sample_fixed_points(arr, 256)
                success.append(1)
            arrays.append(arr)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return

    try:
        concatenated = np.stack(arrays)
        success = np.array(success)
    except Exception as e:
        print(f"Error concatenating arrays in {contact_folder}: {e}")
        return

    # Save the concatenated array as contact_pc.npy
    output_file = os.path.join(contact_folder, 'contact_pc_all.npy')
    success_file = os.path.join(contact_folder, 'success.npy')
    try:
        np.save(output_file, concatenated)
        np.save(success_file, success)
        print(f"Saved concatenated array to {output_file}.")
    except Exception as e:
        print(f"Error saving concatenated array to {output_file}: {e}")

def main():
    # parser = argparse.ArgumentParser(description="Concatenate contact_pc files for each object.")
    # parser.add_argument('--root-dir', type=str, default="grasp_data", help="Root directory of the dataset.")
    # parser.add_argument('--num-workers', type=int, default=48, help="Number of worker processes.")
    # parser.add_argument('--hand', type=str, required=True, help="Hand type to process.")
    # args = parser.parse_args()

    file_dir = os.path.dirname(__file__)
    root_dir = pjoin(file_dir, '../..', "data/grasp_data")
    num_workers = os.cpu_count()
    func_args = []

    if not os.path.isdir(root_dir):
        print(f"Error: {root_dir} directory not found.")
        return

    # Traverse each hand and then each object directory
    for hand in ['Allegro', 'HumanHand', 'shadow_hand']:
        hand_path = os.path.join(root_dir, hand)
        if os.path.isdir(hand_path):
            for obj in os.listdir(hand_path):
                object_path = os.path.join(hand_path, obj)
                if os.path.isdir(object_path):
                    func_args.append((object_path, hand))

    if not func_args:
        print("No object directories found.")
        return

    # Use multiprocessing to process each object directory
    with Pool(processes=num_workers) as pool:
        pool.map(process_object_dir, func_args)

if __name__ == '__main__':
    main()