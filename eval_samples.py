import pickle
import glob
import numpy as np
import torch
from tqdm import tqdm
from chamfer_distance import ChamferDistance as chamfer_dist
from utils.utils_gwh import monte_carlo_iou_1to1_6d_chunked

def thresholded_bidirectional_chamfer(A: torch.Tensor,
                                      B: torch.Tensor,
                                      O: torch.Tensor,
                                      threshold: float = 0.1,
                                      use_cuda: bool = False):
    """
    A: (N,3) tensor
    B: (M,3) tensor
    O: (P,3) tensor
    threshold: distance cutoff for selecting points in O
    Returns: bidirectional Chamfer loss between O_sel_A and O_sel_B
    """
    device = torch.device('cuda') if (use_cuda and torch.cuda.is_available()) else torch.device('cpu')
    A = A.to(device)
    B = B.to(device)
    O = O.to(device)

    # add batch dim
    A_b = A.unsqueeze(0)      # (1, N, 3)
    B_b = B.unsqueeze(0)      # (1, M, 3)
    O_b = O.unsqueeze(0)      # (1, P, 3)

    chamfer = chamfer_dist().to(device)

    # forward distances: O -> A
    dist1_OA, _, idx1_OA, _ = chamfer(O_b, A_b)
    # forward distances: O -> B
    dist1_OB, _, idx1_OB, _ = chamfer(O_b, B_b)

    # remove batch dimension: (P,)
    dist1_OA = dist1_OA.squeeze(0)
    dist1_OB = dist1_OB.squeeze(0)

    # threshold to select subsets of O
    mask_A = dist1_OA < threshold
    mask_B = dist1_OB < threshold

    O_sel_A = O[mask_A]  # (P_A, 3)
    O_sel_B = O[mask_B]  # (P_B, 3)

    if O_sel_A.shape[0] == 0 or O_sel_B.shape[0] == 0:
        # raise ValueError("No points selected under the given threshold. "
        #                  f"Got {O_sel_A.shape[0]} and {O_sel_B.shape[0]} points.")
        return torch.tensor(0.0, device=device)

    # now compute bidirectional Chamfer between O_sel_A and O_sel_B
    Oa_b = O_sel_A.unsqueeze(0)  # (1, P_A, 3)
    Ob_b = O_sel_B.unsqueeze(0)  # (1, P_B, 3)

    dist1_sel, dist2_sel, idx1_sel, idx2_sel = chamfer(Oa_b, Ob_b)

    # bidirectional loss = mean of both forward/backward
    loss = dist1_sel.mean() + dist2_sel.mean()
    return loss

def main():
    # sample_files = glob.glob("./logs/diffusion_ddp/mgg/*/test_output*/samples.pkl")
    # sample_files = glob.glob("./logs/baselines/rfp/*/test_output/samples_1.pkl")
    sample_files = glob.glob("./logs/diffusion_ddp/mgg/*/test_output/samples.pkl")
    for sample_file in sample_files:
        print(f"Processing {sample_file}")
        with open(sample_file, 'rb') as f:
            out_dict = pickle.load(f)
        
        write_out_dict = False
        if len(out_dict['sample_iou']) > 0:
            print(f"Skipping {sample_file} as it already contains IoU values.")
            print(f"Method: {out_dict['method']}, Mean std: {out_dict['sample_std']}")
            print(f"Method: {out_dict['method']}, Mean IoU: {out_dict['sample_iou']['mean']}")
        else:
            all_ious = []
            for object_id, hulls in tqdm(out_dict['sample_hull'].items()):
                hulls_1, hulls_2 = hulls
                object_iou = np.zeros((out_dict['sample_qpos'][object_id].shape[0],), dtype=np.float32)
                iou = monte_carlo_iou_1to1_6d_chunked(hulls_1, hulls_2, samples=500_000,
                                                    chunk_size=100_000)
                iou_np = iou.cpu().numpy().squeeze()
                object_iou[:iou_np.shape[0]] = iou_np
                out_dict['sample_iou'][object_id] = object_iou

                all_ious.append(object_iou)
            all_ious = np.concatenate(all_ious, axis=0)
            print(f"Method: {out_dict['method']}, Mean std: {out_dict['sample_std']}")
            print(f"Method: {out_dict['method']}, Mean IoU: {np.mean(all_ious)}")
            out_dict['sample_iou']['mean'] = np.mean(all_ious)

            write_out_dict = True
            with open(sample_file, 'wb') as f:
                pickle.dump(out_dict, f)
        
        if 'contact_consistency' in out_dict:
            print(f"Method: {out_dict['method']}, Contact Consistency: {out_dict['contact_consistency']}")
        else:
            hand1_pc = out_dict['hand1_pc']
            hand2_pc = out_dict['hand2_pc']
            obj_pc = out_dict['obj_pc']
            contact_consistency = []
            for object_id in tqdm(out_dict['hand1_pc'].keys()):
                hand1_pc_np = hand1_pc[object_id]
                hand2_pc_np = hand2_pc[object_id]
                obj_pc_tensor = torch.from_numpy(obj_pc[object_id]).float()

                # Compute contact consistency
                for i in range(hand1_pc_np.shape[0]):
                    hand1 = torch.from_numpy(hand1_pc_np[i]).float()
                    hand2 = torch.from_numpy(hand2_pc_np[i]).float()

                    # Compute bidirectional Chamfer distance
                    item_contact = thresholded_bidirectional_chamfer(
                            hand1, hand2, obj_pc_tensor, threshold=0.008, use_cuda=torch.cuda.is_available()
                        ).item()
                    if item_contact > 0:
                        contact_consistency.append(item_contact)
            out_dict['contact_consistency'] = np.mean(contact_consistency)
            print(f"Method: {out_dict['method']}, Contact Consistency: {out_dict['contact_consistency']}")
            write_out_dict = True

        if write_out_dict:
            with open(sample_file, 'wb') as f:
                pickle.dump(out_dict, f)

if __name__ == '__main__':
    main()
