import os
import sys
import types

sys.path.append(os.getcwd())

import gc
import yaml
import pickle
import argparse
from loguru import logger

from isaacgym import gymapi, gymutil, gymtorch
import torch
import random
import numpy as np
import pickle as pkl
import trimesh as tm
from utils.rot6d import row6d_to_col6d
import scipy
from scipy.spatial import ConvexHull
# 1) Make an empty module under the missing name
mod = types.ModuleType('scipy.spatial._qhull')

# 2) Point its ConvexHull attribute at the real class
mod.ConvexHull = scipy.spatial.ConvexHull

# 3) Stick it into sys.modules so pickle.find_class will see it
sys.modules['scipy.spatial._qhull'] = mod

from tqdm import tqdm

# scipy.spatial._qhull
# sys.modules['scipy.spatial._qhull'] = types.ModuleType('scipy.spatial._qhull')

def set_global_seed(seed: int) -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Test Scripts of Grasp Generation')
    parser.add_argument('--robot_name', default='allegro_right', type=str)
    parser.add_argument('--stability_config', type=str,
                        default='envs/tasks/grasp_test_force.yaml',
                        help='stability config file path')
    parser.add_argument('--eval_dir', type=str, required=True,
                        help='evaluation directory path for saving results')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='run all on cuda')
    parser.add_argument('--onscreen', action='store_true', default=False,
                        help='run simulator onscreen')

    return parser.parse_args()



def get_sim_param(use_gpu_pipeline=True):
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1. / 60.
    sim_params.num_client_threads = 0
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = True
    sim_params.physx.num_subscenes = 0
    sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
    sim_params.use_gpu_pipeline = use_gpu_pipeline
    sim_params.physx.num_threads = 0
    return sim_params


def stability_tester(args: argparse.Namespace) -> dict:
    with open(args.stability_config) as f:
        stability_config = yaml.safe_load(f)
    sim_params = get_sim_param(use_gpu_pipeline=args.device != "cpu")
    sim_headless = not args.onscreen

    if args.robot_name.lower() == 'allegro_right':
        from envs.tasks.grasp_test_force_allegro import IsaacGraspTestForce_allegro as IsaacGraspTestForce
    elif args.robot_name.lower() == 'shadowhand_nowrist':
        from envs.tasks.grasp_test_force_shadowhand_nowrist import IsaacGraspTestForce_shadowhand_nowrist as IsaacGraspTestForce
    else:
        raise ValueError(f'Robot {args.robot_name} is not supported!')

    # load generated grasp results here
    grasps = pickle.load(open(os.path.join(args.eval_dir, 'samples.pkl'), 'rb'))

    isaac_env = None
    results = {}
    across_all_cases = 0
    across_all_succ = 0
    if 'scale_list' in grasps.keys():
        object_scales = grasps['scale_list']
    else:
        obj_example, grasps_example = next(iter(grasps['sample_qpos'].items()))
        object_scales = [1. for _ in range(len(grasps_example))]
    
    for object_code in tqdm(list(grasps['sample_qpos'].keys())):
        logger.info(f'Stability test for [{object_code}]')
        q_generated = grasps['sample_qpos'][object_code]
        object_scales = [1. for _ in range(len(q_generated))]
        # import pdb;pdb.set_trace()
        q_generated = torch.tensor(q_generated, device=args.device).to(torch.float32)
        q_generated[:, 3:9] = row6d_to_col6d(q_generated[:, 3:9])
        if "shadow" in args.robot_name.lower():
            q_generated[:, 9:] = q_generated[:, 9:][:, [0, 1, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16, 4, 5, 6, 7, 8, 17, 18, 19, 20, 21]]

        if "allegro" in args.robot_name.lower():
            isaac_env = IsaacGraspTestForce(stability_config, sim_params, gymapi.SIM_PHYSX,
                                            args.device, 0, headless=sim_headless, init_opt_q=q_generated,
                                            object_name=object_code, object_scales=object_scales, fix_object=False,robot=args.robot_name.lower(),
                                            mesh_path='./data/mgg_pc/objects/obj')
        elif "shadow" in args.robot_name.lower() or "human" in args.robot_name.lower():
            object_volume = tm.load(os.path.join('./data/mgg_pc/objects/obj', f'{object_code}.obj')).volume
            isaac_env = IsaacGraspTestForce(stability_config, sim_params, gymapi.SIM_PHYSX,
                                            args.device, 0, headless=sim_headless, init_opt_q=q_generated,
                                            object_name=object_code, object_volume=object_volume, fix_object=False)
        else:
            raise ValueError(f'Robot {args.robot_name} is not supported!')
        
        succ_grasp_object = isaac_env.push_object().detach().cpu().numpy()
        results[object_code] = succ_grasp_object.tolist()
        logger.info(
            f'Success rate of [{object_code}]: {int(succ_grasp_object.sum())} / {int(succ_grasp_object.shape[0])}')
        
        across_all_succ += int(succ_grasp_object.sum())
        across_all_cases += int(succ_grasp_object.shape[0])
        if isaac_env is not None:
            del isaac_env
            gc.collect()

    logger.info(f'**Success Rate** across all objects: {across_all_succ} / {across_all_cases}, succ rate: {across_all_succ / across_all_cases}')
    logger.info(f"**GWH_IOU** across all objects: {grasps['sample_iou']['mean']}")
    logger.info(f"**STD** across all objects: {grasps['sample_std']}")
    with open(f'{args.eval_dir}/succ.pickle', 'wb') as f:
        pickle.dump(results, f)

    return results

def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    np.random.seed(10)

    logger.add(args.eval_dir + '/evaluation_right.log')
    logger.info(f'Evaluation directory: {args.eval_dir}')

    logger.info('Start evaluating..')

    stability_results = stability_tester(args)

    logger.info('End evaluating..')


if __name__ == '__main__':
    main()