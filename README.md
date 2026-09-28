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
> **Note:** These steps were reconstructed from the actual setup work and have not been verified by running
> them from start to finish. Some commands were run in a different form (e.g. as one-off Python snippets),
> and the `grasp2grasp` (Isaac Gym) environment already existed and was only extended.

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
A Python 3.8 env with PyTorch (2.4.1+cu118 was used). PyTorch cannot run on the GPU here, so the torch side
runs on CPU (`--device cpu`), while PhysX still simulates on the GPU.
```
conda activate grasp2grasp
# Isaac Gym Preview 4, extracted to ./IsaacGym_Preview_4_Package
echo "$PWD/IsaacGym_Preview_4_Package/isaacgym/python" > $(python -c "import site;print(site.getsitepackages()[0])")/isaacgym_local.pth
pip install urdf-parser-py plotly transformations transforms3d ninja ipdb pyarrow loguru trimesh "pytorch-kinematics==0.5.6"
```
Isaac Gym uses `np.float` too. Patch it:
```
sed -i 's/dtype=np\.float,/dtype=float,/' IsaacGym_Preview_4_Package/isaacgym/python/isaacgym/torch_utils.py
```
`ninja` must be on `PATH` (Isaac Gym compiles `gymtorch` on first import), and Isaac Gym needs the env's
`libpython3.8` on `LD_LIBRARY_PATH`:
```
export PATH=$CONDA_PREFIX/bin:$PATH LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```
The repository does not ship object URDFs. Generate a minimal URDF (plus a symlink to the `.obj`) for each object:
```
python grasp_test/make_object_urdfs.py --mesh_dir data/mgg_pc/objects/obj --out_dir grasp_test/data/mgg_pc/objects/obj
```
To check the simulation alone, without running the model, `grasp_test/make_gt_samples.py` builds a
`samples.pkl` from ground-truth dataset grasps:
```
python grasp_test/make_gt_samples.py --hand Allegro --out_dir logs/isaac_gt_test
cd grasp_test && python isaac_test_right.py --robot_name allegro_right --eval_dir ../logs/isaac_gt_test --device cpu
```
Both `allegro_right` and `shadowhand_nowrist` support `--device cpu`. For the Shadow hand, use
`--hand shadow_hand` and `--robot_name shadowhand_nowrist`. The Shadow task convex-decomposes each object
with VHACD the first time it loads it (about a minute per object, cached in `~/.isaacgym/vhacd`), while the
Allegro task uses a single convex hull of the object.

### 3. Inference with the pretrained Human→Allegro checkpoints
Run these from the repository root in the `g2g` env with the CUDA variables above exported.

**Checkpoints.** Download the [VAE checkpoints](https://drive.google.com/drive/folders/1gtcLW3iFDjiezBFYMq_f6Wpt1hQr6U8d?usp=drive_link),
the [SB checkpoints](https://drive.google.com/drive/folders/1dVmSHcLdnyqgeqBjfQFUCIiHEg0wagFm?usp=drive_link) and the
[LION checkpoint](https://drive.google.com/drive/folders/1pDfkBD0EFCP-L__HfxpcphdSNVf4U2pO?usp=drive_link) (e.g. with `gdown --folder <url>`).
`config/mgg/sample_human_allegro_gwh.json` expects:
```
logs/pretrained_ae/ae_allegro_pvcnn_kl/model_best_test.pth
logs/mgg/mgg/sbfm_human_allegro_gwh/checkpoints/model_epochepoch=00999.ckpt
logs/lion/aeb159h_hvae_lion_B32/{cfg.yml,checkpoints/epoch_5999_iters_1667999.pt}
```
`dataset/scripts/preprocess_latent.py` loads VAE checkpoints from `dataset/logs/autoencoder/mgg/<file_name>/model_best_test.pth`:
```
for h in allegro human; do
  mkdir -p dataset/logs/autoencoder/mgg/ae_${h}_pvcnn_kl
  ln -sfn $PWD/logs/pretrained_ae/ae_${h}_pvcnn_kl/model_best_test.pth dataset/logs/autoencoder/mgg/ae_${h}_pvcnn_kl/
done
```

**Split.** The test dataset reads `data/grasp_data/HumanHand/split.json`. If only the Allegro split exists,
derive it from that one so both hands use the same objects:
```
python -c "
import json
s = json.load(open('data/grasp_data/Allegro/split.json'))
json.dump({k: [p.replace('/Allegro/', '/HumanHand/') for p in v] for k, v in s.items()},
          open('data/grasp_data/HumanHand/split.json', 'w'), indent=4)"
python -c "import json; print('\n'.join(p.split('/')[-1] for p in json.load(open('data/grasp_data/Allegro/split.json'))['test']))" > test_objs.txt
```
A split generated locally may not match the one the checkpoints were trained with, so "test" objects may
have been seen during training.

**Preprocessing (test objects only).** `--objects_file` restricts the human-hand preprocessing to the listed
objects (32 test objects took under an hour on 20 cores, instead of days for the full dataset). Then consolidate the contact
points for both hands:
```
(cd dataset/preproc && python mgg_to_pc_parallel_human.py --objects_file ../../test_objs.txt)
(cd dataset/preproc && python -c "
from multiprocessing import Pool
from process_contact import process_object_dir
objs = open('../../test_objs.txt').read().split()
with Pool(8) as p:
    p.map(process_object_dir, [(f'../../data/grasp_data/{h}/{o}', h) for h in ['Allegro', 'HumanHand'] for o in objs])")
```

**Features.** Object latents are saved to `data/mgg_pc/objects/object_pc`, but the dataset reads
`data/grasp_data/object_pc`, so link them. `--root_dir` must match the path prefix stored in `split.json`
(`./data/grasp_data`), and `CUDA_VISIBLE_DEVICES` overrides the script's default GPU index of 1:
```
ln -sfn ../mgg_pc/objects/object_pc data/grasp_data/object_pc
(cd dataset/scripts && python save_mgg_pc_latent.py)
for h in allegro human; do
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python dataset/scripts/preprocess_latent.py \
      --config config/mgg/ae_${h}_pvcnn.json --root_dir ./data/grasp_data --splits test
done
```

**Sample and evaluate.** `sample.py` writes `logs/diffusion_ddp/mgg/sbfm_human_allegro_gwh/test_output/samples.pkl`
(about 10 GB for 32 objects × 32 samples). On a 32 GB GPU, `eval_samples.py` needs `expandable_segments`
to avoid running out of memory:
```
python sample.py --config config/mgg/sample_human_allegro_gwh.json
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python eval_samples.py
```

**Isaac Gym.** Copy only the grasp poses into a small `samples.pkl` (the full file is too large to load in
the Isaac env), then switch to the `grasp2grasp` env:
```
mkdir -p logs/isaac_h2a
python -c "
import pickle, numpy as np
d = pickle.load(open('logs/diffusion_ddp/mgg/sbfm_human_allegro_gwh/test_output/samples.pkl', 'rb'))
pickle.dump({'method': d['method'], 'sample_qpos': {k: np.asarray(v) for k, v in d['sample_qpos'].items()},
             'sample_iou': {'mean': float(d['sample_iou']['mean'])}, 'sample_std': float(d['sample_std'])},
            open('logs/isaac_h2a/samples.pkl', 'wb'))"
conda activate grasp2grasp
cd grasp_test && python isaac_test_right.py --robot_name allegro_right --eval_dir ../logs/isaac_h2a --device cpu
```

### 4. Inference with the pretrained Human→Shadow checkpoints
This reuses the human-hand data, the object latents and `test_objs.txt` from step 3; only the Shadow side is
new. `config/mgg/sample_human_shadow_gwh.json` expects:
```
logs/pretrained_ae/ae_shadow_pvcnn_kl/model_best_test.pth
logs/mgg/mgg/sbfm_human_shadow_gwh/checkpoints/model_epochepoch=00999.ckpt
```
Link the Shadow VAE checkpoint for `preprocess_latent.py`, and derive the Shadow split from the human one:
```
mkdir -p dataset/logs/autoencoder/mgg/ae_shadow_pvcnn_kl
ln -sfn $PWD/logs/pretrained_ae/ae_shadow_pvcnn_kl/model_best_test.pth dataset/logs/autoencoder/mgg/ae_shadow_pvcnn_kl/
mkdir -p data/grasp_data/shadow_hand
python -c "
import json
s = json.load(open('data/grasp_data/HumanHand/split.json'))
json.dump({k: [p.replace('/HumanHand/', '/shadow_hand/') for p in v] for k, v in s.items()},
          open('data/grasp_data/shadow_hand/split.json', 'w'), indent=4)"
```
Preprocess the Shadow hand for the test objects (32 objects took about 12 minutes on 20 cores), consolidate
contact points and save the VAE features:
```
(cd dataset/preproc && python mgg_to_pc_parallel_shadow.py --objects_file ../../test_objs.txt)
(cd dataset/preproc && python -c "
from multiprocessing import Pool
from process_contact import process_object_dir
objs = open('../../test_objs.txt').read().split()
with Pool(8) as p:
    p.map(process_object_dir, [(f'../../data/grasp_data/shadow_hand/{o}', 'shadow_hand') for o in objs])")
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python dataset/scripts/preprocess_latent.py \
    --config config/mgg/ae_shadow_pvcnn.json --root_dir ./data/grasp_data --splits test
```
Sample and evaluate. `eval_samples.py` evaluates every `samples.pkl` under `logs/diffusion_ddp/mgg/` and
skips the ones that already have IoU values:
```
python sample.py --config config/mgg/sample_human_shadow_gwh.json
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python eval_samples.py
```
Make the small `samples.pkl` as in step 3, but from `sbfm_human_shadow_gwh` into `logs/isaac_h2s`, then:
```
conda activate grasp2grasp
cd grasp_test && python isaac_test_right.py --robot_name shadowhand_nowrist --eval_dir ../logs/isaac_h2s --device cpu
```
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
