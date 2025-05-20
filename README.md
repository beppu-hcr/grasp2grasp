# Submission 25683: Grasp2Grasp: Vision-Based Dexterous Grasp Translation via Schrödinger Bridges


> ⚠️ **IMPORTANT:** This repository is provided **for review only**.  
> **DO NOT DISTRIBUTE** or share these materials.  
> There is **no guarantee** that this code is fully tested or production-ready.

## Table of Contents
- [Introduction](#introduction)
- [Setup](#setup)
- [Data Download](#data-download)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
  - [Train VAE](#train-vae)
  - [Pre-save Features](#pre-save-features)
  - [Train SB Models](#train-sb-models)
- [Usage](#usage)
- [Notes](#notes)

## Introduction
This anonymized repository accompanies Submission 25683: Grasp2Grasp: Vision-Based Dexterous Grasp Translation via Schrödinger Bridges. Use it strictly for peer review.

## Setup
```
conda env create -f environment.yml
conda activate grasp2grasp
```

## Data Download
Visit visit
[https://irvlutd.github.io/MultiGripperGrasp/](https://irvlutd.github.io/MultiGripperGrasp/) and download the MultiGripperGrasp dataset.

## Data Preprocessing
Preprocess the dataset (⚠️ outputs ~1.1 TB) using scripts in `./dataset/preproc`

## Model Training
Train VAE
```
python train_ae.py --config /path/to/vae_config
```

## Pre-save Features
Precompute and save Hand latent representations, GWH, Jacobian information using scripts in `./dataset/scripts` (⚠️ outputs ~2 TB)

## Train SB Models
```
python train_fm_ddp.py --config /path/to/sb_config
```

## Usage
After training, you can evaluate or sample:
```
python sample.py --config /path/to/sb_config
```

## Notes
> Review only: Do NOT distribute.
> No warranty of correctness or completeness.
> Use at your own risk.
