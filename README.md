# Grasp2Grasp: Vision-Based Dexterous Grasp Translation via Schrödinger Bridges (NeurIPS 2025)

## Table of Contents
- [Introduction](#introduction)
- [Setup](#setup)
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
