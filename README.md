# Grasp2Grasp: Vision-Based Dexterous Grasp Translation via Schrödinger Bridges (NeurIPS 2025)

[arXiv](https://arxiv.org/pdf/2506.02489) / [Project Page](https://grasp2grasp.github.io/)

## Table of Contents
- [Introduction](#introduction)
- [Setup](#setup)
- [Setup on RTX 50-series (Blackwell) GPUs](#setup-on-rtx-50-series-blackwell-gpus)
- [Data Download](#data-download)
- [Data Preprocessing](#data-preprocessing)
- [Train VAE](#train-vae)
- [Pre-save Features](#pre-save-features)
- [Train SB Models](#train-sb-models)
- [Usage](#usage)
- [Notes](#notes)

## Introduction
This repository contains official implementation of [Grasp2Grasp: Vision-Based Dexterous Grasp Translation via Schrödinger Bridges](https://arxiv.org/abs/2506.02489).

## Setup
```
conda env create -f environment.yml
conda activate grasp2grasp
```
If you encounter any issue, you might have to build [pytorch3d==0.7.2](https://github.com/facebookresearch/pytorch3d/tree/v0.7.2) and [xformers==0.0.21](https://github.com/facebookresearch/xformers/tree/v0.0.21) from source.

## Setup on RTX 50-series (Blackwell) GPUs
> **Note:** These steps were verified by following them from start to finish in a fresh clone with newly
> created conda environments (Human→Allegro and Human→Shadow on the 32 test objects). The dataset,
> the Isaac Gym archive and the checkpoints were taken from an existing local copy rather than downloaded,
> so the download steps themselves were not tested.

The environment above (PyTorch 1.12 + CUDA 11.3) does not run on sm_120 GPUs such as the RTX 5090.
The steps below were used on an RTX 5090 (driver 595, Ubuntu 24.04) for **inference with the pretrained
checkpoints** (Human→Allegro and Human→Shadow) and the Isaac Gym evaluation. Training was not tested.

Two conda environments are needed, because Isaac Gym only supports Python ≤ 3.8 and no PyTorch build for
Python 3.8 supports sm_120:

| Env | Python | PyTorch | Used for |
|---|---|---|---|
| `g2g` | 3.10 | 2.7.1+cu128 | preprocessing, feature saving, `sample.py`, `eval_samples.py` |
| `grasp2grasp` | 3.8 | 2.4.1+cu118 (used on CPU only) | Isaac Gym evaluation (`grasp_test/`) |

### 1. `g2g` environment (model inference)
CUDA 12.8 (with `nvcc`) is installed into the conda env, so no system CUDA toolkit is needed.
```
conda create -y -n g2g -c nvidia/label/cuda-12.8.1 -c conda-forge \
    python=3.10 cuda-nvcc cuda-cudart-dev cuda-libraries-dev cuda-cccl
conda activate g2g
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

pip install "numpy==1.26.4" easydict einops ipdb loguru natsort open3d pandas pyarrow fastparquet pot \
    "pytorch-kinematics==0.7.5" "pytorch-lightning==1.9.5" "lightning-bolts==0.7.0" scipy torchsde \
    fast-simplification rtree ninja pyyaml tqdm matplotlib "setuptools<70" wheel \
    "trimesh==4.6.4" "libigl==2.5.1" "torch_geometric==2.6.1"
pip install pyg_lib torch_cluster -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
pip install --no-deps urdfpy pycollada lxml "networkx<3.5"
pip install --no-deps xformers==0.0.31.post1 --index-url https://download.pytorch.org/whl/cu128
```
Pinned versions matter:
- `trimesh==4.6.4`: trimesh 5 removed `Trimesh.remove_duplicate_faces`, used by `utils/utils_gwh.py`.
- `libigl==2.5.1`: libigl 2.6 changed `igl.signed_distance` (no `return_normals`), used by `utils/utils_gwh.py`.
- `torch_geometric==2.6.1`: 2.8 requires `pyg-lib>=0.6` for `fps`, which has no PyTorch 2.7 wheel; 2.6.1 falls back to `torch_cluster`.

urdfpy uses `np.float`, which NumPy ≥ 1.24 removed. Patch the installed copy:
```
sed -i 's/np\.float\b/np.float64/g' $CONDA_PREFIX/lib/python3.10/site-packages/urdfpy/urdf.py
```

CUDA extensions (pytorch3d, chamfer_distance, and the pvcnn / LION backends that are JIT-compiled on first
import) need these variables. Export them in every shell that runs the code:
```
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST=12.0
```
Then build pytorch3d and chamfer_distance from source:
```
FORCE_CUDA=1 MAX_JOBS=16 pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"
pip install --no-build-isolation "git+https://github.com/otaheri/chamfer_distance"
```
xformers' `memory_efficient_attention` fails on sm_120 with fp16 inputs (it dispatches to the Hopper
FlashAttention-3 kernel), but works with fp32, which is what this code uses.

### 2. `grasp2grasp` environment (Isaac Gym)
PyTorch cannot run on the GPU in this env, so the torch side runs on CPU (`--device cpu`), while PhysX still
simulates on the GPU. Create the env with PyTorch 2.4.1+cu118:
```
conda create -y -n grasp2grasp python=3.8
conda activate grasp2grasp
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```
Download [Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym). The archive unpacks to `isaacgym/`;
put it under `IsaacGym_Preview_4_Package/` in the repository root, then register it and install the rest:
```
mkdir IsaacGym_Preview_4_Package && tar xf IsaacGym_Preview_4_Package.tar.gz -C IsaacGym_Preview_4_Package
echo "$PWD/IsaacGym_Preview_4_Package/isaacgym/python" > $(python -c "import site;print(site.getsitepackages()[0])")/isaacgym_local.pth
pip install urdf-parser-py plotly transformations transforms3d ninja ipdb pyarrow loguru trimesh pillow "pytorch-kinematics==0.5.6"
```
`pillow` is needed because trimesh loads the textured object meshes. Isaac Gym uses `np.float` too. Patch it:
```
sed -i 's/dtype=np\.float,/dtype=float,/' IsaacGym_Preview_4_Package/isaacgym/python/isaacgym/torch_utils.py
```
`ninja` must be on `PATH` (Isaac Gym compiles `gymtorch` on first import), and Isaac Gym needs the env's
`libpython3.8` on `LD_LIBRARY_PATH`:
```
export PATH=$CONDA_PREFIX/bin:$PATH LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### 3. Data preparation (shared by Human→Allegro and Human→Shadow)
Run these from the repository root in the `g2g` env with the CUDA variables from step 1 exported.
Times are for 20 CPU cores and an RTX 5090.

**Dataset.** Download the [MultiGripperGrasp dataset](https://utdallas.app.box.com/v/multi-gripper-grasp-data/)
and extract it so that these paths exist (extract `GoogleScannedObjects.zip` and `YCB.zip` inside `Object_Models` too):
```
data/multigripper_grasp_data/Dataset/Object_Models/GoogleScannedObjects/<object>/<object>.urdf
data/multigripper_grasp_data/Dataset/Object_Models/YCB/<object>/{textured.obj,points.xyz}
data/multigripper_grasp_data/Dataset/Object_Models/{mgg_models_ids.txt,GoogleScannedObjects_model_ids.txt,ycb_object_ids.txt}
data/multigripper_grasp_data/Dataset/graspit_grasps/{Allegro,HumanHand,shadow_hand}/<hand>-<object>.json
```

**Object meshes** (about 2 minutes). Writes `data/mgg_pc/objects/{obj,npy,mat}` for all 345 objects.
The output directory must exist first:
```
mkdir -p data/mgg_pc/objects
(cd dataset/preproc && python mgg_parse_objects.py)
```

**Split.** `dataset/splits/mgg_split.json` lists the objects of each split (147 train / 32 val / 32 test /
134 reserved). It is the split used for [`results/results.md`](results/results.md). The data loaders read a
`split.json` per hand, so write those, and the list of test objects:
```
python -c "
import json, os
s = json.load(open('dataset/splits/mgg_split.json'))
for hand in ['Allegro', 'HumanHand', 'shadow_hand']:
    os.makedirs(f'data/grasp_data/{hand}', exist_ok=True)
    json.dump({k: [f'./data/grasp_data/{hand}/{o}' for o in v] for k, v in s.items()},
              open(f'data/grasp_data/{hand}/split.json', 'w'), indent=4)
open('test_objs.txt', 'w').write('\n'.join(s['test']) + '\n')"
```
This split was generated locally and may not match the one the checkpoints were trained with, so "test"
objects may have been seen during training.

**Hand point clouds (test objects only).** `--objects_file` restricts preprocessing to the listed objects.
For the 32 test objects this took about 20 minutes for Allegro and 11 minutes each for the human and Shadow
hands, instead of days for the full dataset. Then consolidate the contact points:
```
for s in mgg_to_pc_parallel mgg_to_pc_parallel_human mgg_to_pc_parallel_shadow; do
  (cd dataset/preproc && python $s.py --objects_file ../../test_objs.txt)
done
(cd dataset/preproc && python -c "
from multiprocessing import Pool
from process_contact import process_object_dir
objs = open('../../test_objs.txt').read().split()
with Pool(8) as p:
    p.map(process_object_dir, [(f'../../data/grasp_data/{h}/{o}', h) for h in ['Allegro', 'HumanHand', 'shadow_hand'] for o in objs])")
```
The object point clouds are sampled at random, so re-running this can change the number of grasps kept per
object by a few (a grasp is dropped when it has no contact points).

**Checkpoints.** Download the [VAE checkpoints](https://drive.google.com/drive/folders/1gtcLW3iFDjiezBFYMq_f6Wpt1hQr6U8d?usp=drive_link),
the [SB checkpoints](https://drive.google.com/drive/folders/1dVmSHcLdnyqgeqBjfQFUCIiHEg0wagFm?usp=drive_link) and the
[LION checkpoint](https://drive.google.com/drive/folders/1pDfkBD0EFCP-L__HfxpcphdSNVf4U2pO?usp=drive_link) (e.g. with `gdown --folder <url>`)
and place them as follows (`config/mgg/sample_human_{allegro,shadow}_gwh.json` and `dataset/scripts/save_mgg_pc_latent.py` use these paths):
```
logs/pretrained_ae/ae_{allegro,human,shadow}_pvcnn_kl/model_best_test.pth
logs/mgg/mgg/sbfm_human_allegro_gwh/checkpoints/model_epochepoch=00999.ckpt
logs/mgg/mgg/sbfm_human_shadow_gwh/checkpoints/model_epochepoch=00999.ckpt
logs/lion/aeb159h_hvae_lion_B32/{cfg.yml,checkpoints/epoch_5999_iters_1667999.pt}
```
`dataset/scripts/preprocess_latent.py` loads VAE checkpoints from `dataset/logs/autoencoder/mgg/<file_name>/model_best_test.pth`:
```
for h in allegro human shadow; do
  mkdir -p dataset/logs/autoencoder/mgg/ae_${h}_pvcnn_kl
  ln -sfn $PWD/logs/pretrained_ae/ae_${h}_pvcnn_kl/model_best_test.pth dataset/logs/autoencoder/mgg/ae_${h}_pvcnn_kl/
done
```

**Features** (about 12 minutes). Object latents are saved to `data/mgg_pc/objects/object_pc`, but the dataset
reads `data/grasp_data/object_pc`, so link them. `--root_dir` must match the path prefix stored in
`split.json` (`./data/grasp_data`), and `CUDA_VISIBLE_DEVICES` overrides the script's default GPU index of 1:
```
ln -sfn ../mgg_pc/objects/object_pc data/grasp_data/object_pc
(cd dataset/scripts && python save_mgg_pc_latent.py)
for h in allegro human shadow; do
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python dataset/scripts/preprocess_latent.py \
      --config config/mgg/ae_${h}_pvcnn.json --root_dir ./data/grasp_data --splits test
done
```

**Object URDFs for Isaac Gym.** The repository does not ship them. Generate a minimal URDF (plus a symlink to
the `.obj`) for each object. The first command writes URDFs with `mass=0.1`, used by the original settings;
the second writes URDFs without mass, so Isaac Gym computes the mass from density 10,000, used by the
recommended settings (step 5):
```
python grasp_test/make_object_urdfs.py --mesh_dir data/mgg_pc/objects/obj --out_dir grasp_test/data/mgg_pc/objects/obj
python grasp_test/make_object_urdfs.py --mesh_dir data/mgg_pc/objects/obj --out_dir grasp_test/data/mgg_pc/objects/obj_density --no_mass
```

To check the simulation alone, without running the model, `grasp_test/make_gt_samples.py` builds a
`samples.pkl` from ground-truth dataset grasps. It needs pandas and pytorch3d, so run it in the `g2g` env,
then run the simulation in the `grasp2grasp` env:
```
python grasp_test/make_gt_samples.py --hand Allegro --out_dir logs/isaac_gt_test
conda activate grasp2grasp
cd grasp_test && python isaac_test_right.py --robot_name allegro_right --eval_dir ../logs/isaac_gt_test --device cpu \
    --stability_config envs/tasks/grasp_test_force_paper.yaml
```
Both `allegro_right` and `shadowhand_nowrist` support `--device cpu`. For the Shadow hand, use
`--hand shadow_hand` and `--robot_name shadowhand_nowrist`. The Shadow task convex-decomposes each object
with VHACD the first time it loads it (about a minute per object, cached in `~/.isaacgym/vhacd`), while the
Allegro task uses a single convex hull of the object.

### 4. Inference with the pretrained checkpoints
In the `g2g` env. `sample.py` writes `logs/diffusion_ddp/mgg/<method>/test_output/samples.pkl` (about 10 GB
for H→A and 8 GB for H→S; about 16 minutes each). `eval_samples.py` evaluates every `samples.pkl` under
`logs/diffusion_ddp/mgg/` and skips the ones that already have IoU values; on a 32 GB GPU it needs
`expandable_segments` to avoid running out of memory:
```
python sample.py --config config/mgg/sample_human_allegro_gwh.json
python sample.py --config config/mgg/sample_human_shadow_gwh.json
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python eval_samples.py
```

### 5. Isaac Gym evaluation
Copy only the grasp poses into a small `samples.pkl` per direction (the full files are too large to load in
the Isaac env). In the `g2g` env:
```
for m in allegro:h2a shadow:h2s; do
  src=${m%%:*}; dst=${m##*:}; mkdir -p logs/isaac_$dst
  python -c "
import pickle, numpy as np
d = pickle.load(open('logs/diffusion_ddp/mgg/sbfm_human_${src}_gwh/test_output/samples.pkl', 'rb'))
pickle.dump({'method': d['method'], 'sample_qpos': {k: np.asarray(v) for k, v in d['sample_qpos'].items()},
             'sample_iou': {'mean': float(d['sample_iou']['mean'])}, 'sample_std': float(d['sample_std'])},
            open('logs/isaac_$dst/samples.pkl', 'wb'))"
done
```
Then evaluate in the `grasp2grasp` env. The recommended settings, `envs/tasks/grasp_test_force_paper.yaml`,
follow the paper where it is explicit (6 directions × 60 steps, success if the object moves < 2 cm in all six,
friction 10, object density 10,000) and keep the authors' code elsewhere, including their push force
(`5000 × mesh volume`, i.e. density 10,000 × 0.5 m/s²). They need the URDFs without mass (`obj_density`, see
step 3) and use VHACD collision for both hands. This takes about 7 minutes for H→A and 35 minutes for H→S:
```
conda activate grasp2grasp
cd grasp_test
python isaac_test_right.py --robot_name allegro_right --eval_dir ../logs/isaac_h2a --device cpu \
    --stability_config envs/tasks/grasp_test_force_paper.yaml
python isaac_test_right.py --robot_name shadowhand_nowrist --eval_dir ../logs/isaac_h2s --device cpu \
    --stability_config envs/tasks/grasp_test_force_paper.yaml
```
The script reads `samples.pkl` from `--eval_dir` and writes `succ.pickle` and `evaluation_right.log` there.

Other settings, as options (pass them with `--stability_config` instead):
- `envs/tasks/grasp_test_force.yaml`, the default when `--stability_config` is omitted: the original
  `grasp_test` settings. The objects keep `mass=0.1` from the default URDFs, so the push reaches up to about
  350 m/s² instead of 0.5 m/s²; 50 steps per direction; the Allegro task collides with a single convex hull.
- `envs/tasks/grasp_test_force_paper_accel.yaml`: the recommended settings with the push applied as
  `mass × 0.5`, i.e. a uniform 0.5 m/s² for every object as the paper's text states. With the authors'
  force, hollow objects get less (0.14–0.5 m/s²).

The configs differ only in the optional keys `object.urdf_dir`, `object.collision` and
`eval_policy.dynamic.acceleration` (without them, both tasks behave as in the original code) and in the
value of `eval_policy.dynamic.num_steps`.

The results of both directions are summarized in [`results/results.md`](results/results.md).

### Troubleshooting
- **`nvidia-smi` fails after a kernel update.** With Ubuntu's prebuilt driver modules, the module package
  for the new kernel may be missing. Install it and load it (or reboot):
  `sudo apt install linux-modules-nvidia-595-open-generic-hwe-24.04 && sudo modprobe nvidia`.
- **Permission denied under `data/` or `logs/`.** Directories created from a Docker container are owned by
  root. Fix with `sudo chown -R $USER:$USER data dataset/preproc logs`.

## Data Download
Download the [MultiGripperGrasp dataset](https://utdallas.app.box.com/v/multi-gripper-grasp-data/). Place and extract the dataset under `./data/`.

## Data Preprocessing
> ⚠️ **IMPORTANT:** Please reserve at least 1 TB of disk space. Due to the size of the dataset, the preprocessing takes ~2 days to finish on a 48-core CPU.

Preprocess the dataset and generate point cloud observations using:
```
cd dataset/preproc && \
python mgg_parse_objects.py && \
python mgg_to_pc_parallel.py && \
python mgg_to_pc_parallel_human.py && \
python mgg_to_pc_parallel_shadow.py && \
python process_contact.py
```

## Train VAE
Train the VAE using:
```
python train_ae.py --config /path/to/vae_config
```
We provide example config files under `./config/mgg`. You can also download and place the [trained checkpoints](https://drive.google.com/drive/folders/1gtcLW3iFDjiezBFYMq_f6Wpt1hQr6U8d?usp=drive_link) under `./logs`.

## Pre-save Features
Precompute and save object point clouds features:
```
cd dataset/scripts && \
python save_mgg_pc_latent.py
```
You can download the pretrained LION checkpoints [here](https://drive.google.com/drive/folders/1pDfkBD0EFCP-L__HfxpcphdSNVf4U2pO?usp=drive_link) and place under `./logs`.

Precompute and save hand point clouds features:
```
cd dataset/scripts && \
python preprocess_latent.py --config /path/to/vae_config
```
where `/path/to/vae_config` is the config file of the corresponding VAE.

(Optional) Precompute and save grasp GWH:
```
cd dataset/scripts && \
python compute_gwh.py
```

(Optional) We provided the precomputed Jacobian of each grasp [here](https://drive.google.com/drive/folders/1pywlCBnUkQbcb2vUbBa5Mpuozq416Jas?usp=drive_link).

## Train SB Models
Train the SB model using:
```
python train_fm_ddp.py --config /path/to/sb_config
```
We provide an example config file `sbfm_human_allegro.json` under `./config/mgg`. Due to cloud drive size limit, we release the pretrained checkpoints of the `H->A` and `H->S` settings trained with the GWH silimarity metric [here](https://drive.google.com/drive/folders/1dVmSHcLdnyqgeqBjfQFUCIiHEg0wagFm?usp=drive_link).

## Evaluation
After training, you can sample via:
```
python sample.py --config /path/to/sb_config
```
Report the results by:
```
python eval_samples.py
```

## (Optional) Isaac Gym Simulation
Install extra dependencies:
```
pip install urdf-parser-py plotly transformations transforms3d
```
Install [Isaac Gym](https://developer.nvidia.com/isaac-gym) and run:
```
cd grasp_test && python isaac_test_right.py --robot_name <ROBOT_NAME> --eval_dir <PATH_TO_SAMPLE_FOLDER>
```

## Citation

If you find this codebase useful in your research, consider citing:

``` bibtex
@inproceedings{
    zhong2025grasp2grasp,
    title={Grasp2Grasp: Vision-Based Dexterous Grasp Translation via Schr\"odinger Bridges},
    author={Tao Zhong and Jonah Buchanan and Christine Allen-Blanchette},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS)},
    year={2025}
}
```

## Credits

The following repositories are used in this repository, either in close to original form or as an inspiration:

* [GenDexGrasp](https://github.com/tengyu-liu/GenDexGrasp/tree/main)
* [Fast-Grasp'D](https://github.com/dylanturpin/fast-graspd)
* [DexDiffuser](https://github.com/YuLiHN/DexDiffuser)
* [UGG](https://github.com/Jiaxin-Lu/ugg)
* [LION](https://github.com/nv-tlabs/LION)
* [U-ViT](https://github.com/baofff/U-ViT/tree/main)
* [FastGrasp](https://github.com/wuxiaofei01/FastGrasp)
* [TorchCFM](https://github.com/atong01/conditional-flow-matching)

## License

Unless otherwise noted in the submodules, the rest of this repo is licensed under the MIT License. See [LICENSE](LICENSE) for more details.
