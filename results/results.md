# Grasp translation results: Human→Allegro and Human→Shadow

Inference with the authors' pretrained checkpoints on 32 test objects, evaluated with `eval_samples.py` and the
Isaac Gym stability test (`grasp_test/isaac_test_right.py`). Sampling ran on 2026-09-28; the Isaac results
reported as the main results use the evaluation settings matched to the paper
(`grasp_test/envs/tasks/grasp_test_force_paper.yaml`, run on 2026-09-29).

These numbers are not directly comparable to the paper: the test objects, the split and the software versions
differ (see [Caveats](#caveats)).

## Summary

| Metric | H→A | H→S | Paper, Ours<sub>GWH</sub>: H→A | Paper, Ours<sub>GWH</sub>: H→S |
|---|---|---|---|---|
| **Isaac success rate** (paper settings) | **615 / 1,016 = 60.5%** | **398 / 1,009 = 39.4%** | 77.73% | 42.59% |
| 6D GWH IoU | 17.9% | 18.1% | 15.09% | 14.14% |
| Diversity (rad) | 0.307 | 0.202 | 0.293 | 0.197 |
| Contact consistency | 0.00129 | 0.00102 | – | – |

- Paper values are from Table 1 of the paper (Ours<sub>GWH</sub>, the variant whose checkpoints were used here).
- `sample.py` generates 32 samples per object (1,024 per direction); samples whose GWH hull could not be
  computed are dropped, which leaves 1,016 (H→A) and 1,009 (H→S).
- Diversity is the mean standard deviation of the joint angles over all generated samples (`sample_std`, computed
  in `sample.py`). It is not comparable across hands: Allegro has 16 joints and Shadow has 22.
- For comparison, the same grasps under the original `grasp_test` settings scored 436 / 1,016 = 42.9% (H→A) and
  241 / 1,009 = 23.9% (H→S), and with a uniform 0.5 m/s² push (reference setting, below) 581 / 1,016 = 57.2% and
  280 / 1,009 = 27.8%.

## Why the paper settings are the main result

The paper describes the Isaac test as: push the object along ±x, ±y, ±z with a uniform acceleration of 0.5 m/s²
for 60 simulation steps each; a grasp succeeds if the object moves less than 2 cm in all six trials; friction
10; object density 10,000; Isaac Gym's built-in position controller. The original `grasp_test` settings
differed from this in ways that matter, most importantly the object mass (see
[Evaluation settings](#evaluation-settings)).

`grasp_test_force_paper.yaml` follows the paper wherever the paper is explicit and keeps the authors' code
everywhere else. That includes how the push is applied: the authors' code applies a force of
`magnitude_per_volume × mesh volume` with `magnitude_per_volume = 5000`, which is density 10,000 × 0.5 m/s².
It is their implementation of the 0.5 m/s² push, and it gives exactly 0.5 m/s² only when the simulated mass
equals density × mesh volume.

It is also the setting closest to the success rates the paper reports:

| Setting | H→A | H→S |
|---|---|---|
| Paper, Ours<sub>GWH</sub> (Table 1) | 77.73% | 42.59% |
| **Paper settings, authors' force (main)** | **60.5%** | **39.4%** |
| Uniform 0.5 m/s² push (reference) | 57.2% | 27.8% |
| Original `grasp_test` settings | 42.9% | 23.9% |

H→S is within about 3 points of the paper; H→A is still about 17 points lower, for reasons not identified.

## Isaac success rate by object (paper settings)

Successful grasps / grasps evaluated, `grasp_test_force_paper.yaml`:

| Object | H→A | H→S |
|---|---|---|
| 3D_Dollhouse_Happy_Brother | 22/32 | 12/32 |
| 3D_Dollhouse_Lamp | 25/32 | 21/31 |
| ACE_Coffee_Mug_Kristen_16_oz_cup | 24/32 | 18/32 |
| Android_Figure_Panda | 28/32 | 25/32 |
| Asus_80211ac_DualBand_Gigabit_Wireless_Router_RTAC68R | 2/32 | 0/30 |
| BIA_Cordon_Bleu_White_Porcelain_Utensil_Holder_900028 | 24/32 | 19/32 |
| Black_Forest_Fruit_Snacks_28_Pack_Grape | 5/31 | 0/31 |
| Clue_Board_Game_Classic_Edition | 2/31 | 0/32 |
| Cole_Hardware_Deep_Bowl_Good_Earth_1075 | 18/30 | 22/31 |
| Cole_Hardware_Mini_Honey_Dipper | 19/31 | 13/31 |
| Corningware_CW_by_Corningware_3qt_Oblong_Casserole_Dish_Blue | 25/32 | 15/29 |
| Design_Ideas_Drawer_Store_Organizer | 25/31 | 11/32 |
| Ecoforms_Garden_Pot_GP16ATurquois | 25/32 | 32/32 |
| Ecoforms_Pot_Nova_6_Turquoise | 21/32 | 32/32 |
| Grreat_Choice_Dog_Double_Dish_Plastic_Blue | 23/32 | 15/32 |
| Logitech_Ultimate_Ears_Boom_Wireless_Speaker_Night_Black | 22/32 | 0/32 |
| Nikon_1_AW1_w11275mm_Lens_Silver | 31/32 | 19/32 |
| Nintendo_2DS_Crimson_Red | 22/31 | 6/31 |
| Nordic_Ware_Original_Bundt_Pan | 23/32 | 31/32 |
| Ortho_Forward_Facing_CkAW6rL25xH | 7/32 | 3/32 |
| Razer_Abyssus_Ambidextrous_Gaming_Mouse | 18/32 | 15/32 |
| Razer_Blackwidow_Tournament_Edition_Keyboard | 13/32 | 3/32 |
| Razer_Goliathus_Control_Edition_Small_Soft_Gaming_Mouse_Mat | 25/32 | 11/32 |
| Remington_TStudio_Hair_Dryer | 12/32 | 7/32 |
| Sushi_Mat | 25/32 | 11/31 |
| TERREX_FAST_X_GTX | 22/32 | 8/32 |
| Threshold_Hand_Towel_Blue_Medallion_16_x_27 | 8/32 | 2/31 |
| Threshold_Porcelain_Pitcher_White | 10/32 | 9/32 |
| Threshold_Salad_Plate_Square_Rim_Porcelain | 27/32 | 15/31 |
| Timberland_Mens_Earthkeepers_Stormbuck_Lite_Plain_Toe_Oxford | 23/32 | 12/31 |
| W_Lou_z0dkC78niiZ | 13/32 | 5/31 |
| YumYum_D3_Liquid | 26/31 | 6/32 |
| **Total** | **615/1,016** | **398/1,009** |

- The router, the board game and the fruit-snack box almost always fail for both hands; the speaker fails for
  H→S only (22/32 for H→A).
- The Shadow hand holds the pots and the Bundt pan well (31–32/32); under the uniform 0.5 m/s² push these
  drop to 11–20/32 (below), so part of this comes from the weaker push those hollow objects get with the
  authors' force.

## Evaluation settings

| Item | Paper | Original `grasp_test` (`grasp_test_force.yaml`) | **Paper settings (`grasp_test_force_paper.yaml`)** | Reference: uniform 0.5 m/s² (`grasp_test_force_paper_accel.yaml`) |
|---|---|---|---|---|
| Steps per direction | 60 | 50 | 60 | 60 |
| Object mass | density 10,000 | `mass=0.1` from the generated URDFs, which overrides the density | URDFs without mass (`make_object_urdfs.py --no_mass`), so Isaac Gym uses density 10,000 | same as paper settings |
| Push | uniform 0.5 m/s² | force `5000 × mesh volume` | force `5000 × mesh volume` (authors' code) | force `mass × 0.5` |
| Resulting acceleration (32 test objects) | 0.5 m/s² | 0.52–349 m/s² | 0.14–0.50 m/s² | 0.5 m/s² |
| Object collision | not stated | Allegro: single convex hull; Shadow: VHACD | VHACD for both | VHACD for both |
| Success, friction, directions | < 2 cm in all 6; friction 10 | same | same | same |
| Not in the paper (authors' code, unchanged) | | object damping (linear 10, angular 100); finger position control (Allegro: gains 1e10 with targets at the generated pose + 0.2 rad; Shadow: stiffness/damping 400 with the closure-optimized pose); dt 1/60, 2 substeps, 4 position iterations | same | same |

With the authors' force, the acceleration is below 0.5 m/s² for hollow objects: Isaac Gym computes the mass
from the collision shapes, which are larger than the mesh for them (about 0.14–0.2 m/s² for the pots and the
Bundt pan). Per-object values are in `paper_eval/object_accel.csv`.

### Ground-truth grasps, step by step
The same 256 ground-truth grasps per hand (32 test objects × 8 random dataset grasps, from
`grasp_test/make_gt_samples.py`), with one change added at each stage:

| Stage | Settings | Allegro | Shadow |
|---|---|---|---|
| 0 | Original (`grasp_test_force.yaml`) | 95/256 = 37.1% | 41/256 = 16.0% |
| 1 | + URDFs without mass (density 10,000) + 60 steps (`paper_eval/configs/stage1.yaml`) | 176/256 = 68.8% | 90/256 = 35.2% |
| **2** | **+ VHACD for both tasks (`grasp_test_force_paper.yaml`, main)** | **134/256 = 52.3%** | **87/256 = 34.0%** |
| 3 | Reference: + force = mass × 0.5 (`grasp_test_force_paper_accel.yaml`) | 125/256 = 48.8% | 50/256 = 19.5% |

- Stage 0 → 1: the mass fix removes pushes of up to 349 m/s², which failed most large objects.
- Stage 1 → 2: with VHACD, hollow objects get a smaller collision volume, hence a smaller mass and a larger
  acceleration (closer to 0.5 m/s²), and different contacts. The Shadow task already used VHACD.
- Stage 2 → 3: every object is pushed at 0.5 m/s², which mostly strengthens the push on hollow objects.
- The ground-truth grasps are random dataset grasps, including ones that fail, so they are not an upper bound
  for the generated grasps.

## Comparison: original settings and the uniform 0.5 m/s² reference

Per-object success for the same generated grasps under the three settings. Acceleration columns are the push
acceleration in m/s² (the reference setting is 0.5 for every object).

| Object | Accel. original | Accel. paper settings | H→A original | **H→A paper settings** | H→A 0.5 m/s² | H→S original | **H→S paper settings** | H→S 0.5 m/s² |
|---|---|---|---|---|---|---|---|---|
| 3D_Dollhouse_Happy_Brother | 2.24 | 0.44 | 28/32 | **22/32** | 21/32 | 12/32 | **12/32** | 12/32 |
| 3D_Dollhouse_Lamp | 4.44 | 0.47 | 23/32 | **25/32** | 25/32 | 20/31 | **21/31** | 20/31 |
| ACE_Coffee_Mug_Kristen_16_oz_cup | 9.71 | 0.35 | 31/32 | **24/32** | 24/32 | 13/32 | **18/32** | 17/32 |
| Android_Figure_Panda | 5.00 | 0.47 | 30/32 | **28/32** | 28/32 | 25/32 | **25/32** | 25/32 |
| Asus_80211ac_DualBand_Gigabit_Wireless_Router_RTAC68R | 348.55 | 0.48 | 0/32 | **2/32** | 1/32 | 0/30 | **0/30** | 0/30 |
| BIA_Cordon_Bleu_White_Porcelain_Utensil_Holder_900028 | 23.59 | 0.33 | 13/32 | **24/32** | 23/32 | 14/32 | **19/32** | 14/32 |
| Black_Forest_Fruit_Snacks_28_Pack_Grape | 219.15 | 0.50 | 2/31 | **5/31** | 5/31 | 0/31 | **0/31** | 0/31 |
| Clue_Board_Game_Classic_Edition | 286.53 | 0.46 | 0/31 | **2/31** | 1/31 | 0/32 | **0/32** | 0/32 |
| Cole_Hardware_Deep_Bowl_Good_Earth_1075 | 53.88 | 0.34 | 3/30 | **18/30** | 20/30 | 8/31 | **22/31** | 15/31 |
| Cole_Hardware_Mini_Honey_Dipper | 0.52 | 0.49 | 23/31 | **19/31** | 18/31 | 13/31 | **13/31** | 12/31 |
| Corningware_CW_by_Corningware_3qt_Oblong_Casserole_Dish_Blue | 34.46 | 0.32 | 9/32 | **25/32** | 24/32 | 12/29 | **15/29** | 12/29 |
| Design_Ideas_Drawer_Store_Organizer | 40.03 | 0.43 | 9/31 | **25/31** | 25/31 | 7/32 | **11/32** | 9/32 |
| Ecoforms_Garden_Pot_GP16ATurquois | 7.99 | 0.21 | 17/32 | **25/32** | 21/32 | 9/32 | **32/32** | 11/32 |
| Ecoforms_Pot_Nova_6_Turquoise | 10.77 | 0.20 | 13/32 | **21/32** | 22/32 | 13/32 | **32/32** | 11/32 |
| Grreat_Choice_Dog_Double_Dish_Plastic_Blue | 45.09 | 0.31 | 4/32 | **23/32** | 20/32 | 8/32 | **15/32** | 15/32 |
| Logitech_Ultimate_Ears_Boom_Wireless_Speaker_Night_Black | 103.11 | 0.48 | 12/32 | **22/32** | 22/32 | 0/32 | **0/32** | 0/32 |
| Nikon_1_AW1_w11275mm_Lens_Silver | 11.10 | 0.47 | 29/32 | **31/32** | 31/32 | 9/32 | **19/32** | 15/32 |
| Nintendo_2DS_Crimson_Red | 13.27 | 0.44 | 21/31 | **22/31** | 22/31 | 2/31 | **6/31** | 3/31 |
| Nordic_Ware_Original_Bundt_Pan | 16.01 | 0.14 | 6/32 | **23/32** | 20/32 | 20/32 | **31/32** | 20/32 |
| Ortho_Forward_Facing_CkAW6rL25xH | 297.06 | 0.43 | 0/32 | **7/32** | 5/32 | 0/32 | **3/32** | 1/32 |
| Razer_Abyssus_Ambidextrous_Gaming_Mouse | 9.17 | 0.44 | 26/32 | **18/32** | 18/32 | 12/32 | **15/32** | 12/32 |
| Razer_Blackwidow_Tournament_Edition_Keyboard | 78.50 | 0.41 | 8/32 | **13/32** | 11/32 | 3/32 | **3/32** | 2/32 |
| Razer_Goliathus_Control_Edition_Small_Soft_Gaming_Mouse_Mat | 8.62 | 0.31 | 16/32 | **25/32** | 23/32 | 6/32 | **11/32** | 6/32 |
| Remington_TStudio_Hair_Dryer | 58.41 | 0.43 | 7/32 | **12/32** | 12/32 | 2/32 | **7/32** | 6/32 |
| Sushi_Mat | 8.79 | 0.35 | 19/32 | **25/32** | 25/32 | 9/31 | **11/31** | 11/31 |
| TERREX_FAST_X_GTX | 66.92 | 0.34 | 9/32 | **22/32** | 17/32 | 1/32 | **8/32** | 3/32 |
| Threshold_Hand_Towel_Blue_Medallion_16_x_27 | 69.24 | 0.44 | 7/32 | **8/32** | 5/32 | 1/31 | **2/31** | 3/31 |
| Threshold_Porcelain_Pitcher_White | 105.53 | 0.28 | 2/32 | **10/32** | 7/32 | 0/32 | **9/32** | 2/32 |
| Threshold_Salad_Plate_Square_Rim_Porcelain | 14.63 | 0.35 | 15/32 | **27/32** | 26/32 | 8/31 | **15/31** | 10/31 |
| Timberland_Mens_Earthkeepers_Stormbuck_Lite_Plain_Toe_Oxford | 45.12 | 0.33 | 22/32 | **23/32** | 19/32 | 6/31 | **12/31** | 5/31 |
| W_Lou_z0dkC78niiZ | 79.14 | 0.44 | 7/32 | **13/32** | 15/32 | 1/31 | **5/31** | 1/31 |
| YumYum_D3_Liquid | 4.84 | 0.48 | 25/31 | **26/31** | 25/31 | 7/32 | **6/32** | 7/32 |
| **Total** | | | 436/1,016 | **615/1,016** | 581/1,016 | 241/1,009 | **398/1,009** | 280/1,009 |

- Original → paper settings: hollow or heavy objects (bowl, pots, Bundt pan, dog dish, casserole dish) gain
  the most. A few objects drop even though their push became much weaker (e.g. the mug for H→A: 31 → 24,
  acceleration 9.7 → 0.35 m/s²; the Razer mouse: 26 → 18, 9.2 → 0.45 m/s²). The cause was not isolated; the
  collision shape (convex hull → VHACD for the Allegro task) and the longer push (50 → 60 steps) also changed.
- Paper settings → 0.5 m/s²: objects whose acceleration was well below 0.5 drop the most, especially for
  the Shadow hand (pots 32 → 11, Bundt pan 31 → 20). A possible reason, not verified: the Allegro fingers are
  held with gains of 1e10, while the Shadow fingers use stiffness 400 and can be pushed back.

## Run conditions

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 (32 GB), driver 595 |
| OS | Ubuntu 24.04, kernel 7.0.0 |
| Model env (`g2g`) | Python 3.10, PyTorch 2.7.1+cu128 |
| Isaac env (`grasp2grasp`) | Python 3.8, PyTorch 2.4.1+cu118 on CPU (`--device cpu`), Isaac Gym Preview 4 with PhysX on GPU |
| SB checkpoints | `sbfm_human_allegro_gwh`, `sbfm_human_shadow_gwh` (epoch 999, GWH OT cost) |
| VAE checkpoints | `ae_allegro_pvcnn_kl`, `ae_shadow_pvcnn_kl` |
| Configs | `config/mgg/sample_human_allegro_gwh.json`, `config/mgg/sample_human_shadow_gwh.json` |
| Test objects | 32, the `test` split of `dataset/splits/mgg_split.json` |
| Samples | 32 per object, SDE integrated with Euler, `dt=0.005` (`sample.py` defaults), seed 42 |
| IoU | Monte Carlo, 500,000 samples (`eval_samples.py`) |
| Isaac stability test | Main: `grasp_test/envs/tasks/grasp_test_force_paper.yaml` with URDFs from `make_object_urdfs.py --no_mass` |

The setup steps are in the README section "Setup on RTX 50-series (Blackwell) GPUs".

## Caveats
- **Test objects and split.** The paper tests on 34 unseen objects; here the 32 objects of a split generated
  locally (`dataset/splits/mgg_split.json`). If it differs from the split the checkpoints were trained with,
  some "test" objects may have been seen during training.
- **Evaluation settings.** The main results follow the paper where it is explicit and the authors' code
  elsewhere; the push acceleration therefore depends on the object (0.14–0.50 m/s²) instead of a uniform
  0.5 m/s². The two Isaac tasks still differ in finger control (see [Evaluation settings](#evaluation-settings)),
  so the H→A / H→S gap does not directly measure model quality.
- **Object URDFs** are generated by `grasp_test/make_object_urdfs.py` (the repository does not ship them):
  one link with the `.obj` as visual and collision mesh. By default the link has `mass=0.1`, which Isaac Gym
  keeps and which made the original evaluation push objects far harder than the paper; the paper settings
  use URDFs without mass (`--no_mass`).
- **The Shadow task is not deterministic.** Re-running the same grasps can change individual results
  (e.g. 1/4, 2/4 and 3/4 on the same four mug grasps), so Shadow success rates carry a few points of noise.
  The Allegro task reproduced the same results in the one repeat that was tried.
- **CPU torch pipeline.** The Isaac tasks were patched to run torch on CPU (`--device cpu`), because no PyTorch
  build for Python 3.8 supports sm_120. PhysX still runs on the GPU. The authors used the GPU pipeline.
- **Library versions** differ from `environment.yml` (PyTorch 2.7.1 instead of 1.12.1, pytorch3d 0.7.9,
  xformers 0.0.31), and some libraries were pinned or patched to keep the code working; see the README.
- **Data loader fix.** Per-grasp files are named by the original grasp index, which skips grasps dropped in
  preprocessing. The loader now maps metadata row k to the k-th file in sorted order (`dataset/data_utils.py`).

## Files

| Path | Contents |
|---|---|
| `paper_eval/stage2_paper/isaac_{h2a,h2s}/` | **Main results**: generated grasps under the paper settings (`succ.pickle`, `evaluation_right.log`, `run.log`) |
| `paper_eval/stage3_accel/isaac_{h2a,h2s}/` | Reference: generated grasps with a uniform 0.5 m/s² push |
| `isaac_h2a/`, `isaac_h2s/` | Comparison: generated grasps under the original `grasp_test` settings |
| `paper_eval/stage{0,1,2,3}_*/gt_{allegro,shadow}/` | Ground-truth runs per stage (`succ.pickle`, `evaluation_right.log`) |
| `paper_eval/object_accel.csv` | Per-object mesh volume, push force, and Isaac Gym mass / acceleration for each setting |
| `paper_eval/configs/stage1.yaml` | Config of stage 1 (the other stages use the configs in `grasp_test/envs/tasks/`) |

`succ.pickle` holds `{object_id: [bool, ...]}`, one success flag per grasp. Not included because of size:
the full `samples.pkl` files (about 10 GB for H→A and 7.7 GB for H→S, in
`logs/diffusion_ddp/mgg/<method>/test_output/`), which hold the generated grasps, hulls, point clouds and
per-sample IoU.
