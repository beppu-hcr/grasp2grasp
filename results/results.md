# Grasp translation results: Human→Allegro and Human→Shadow

Inference with the authors' pretrained checkpoints on 32 test objects, evaluated with `eval_samples.py`
and the Isaac Gym stability test (`grasp_test/isaac_test_right.py`). Run on 2026-09-28; the Isaac test was
re-run with settings matched to the paper on 2026-09-29 (see
[Evaluation matched to the paper](#evaluation-matched-to-the-paper)).

The first Isaac evaluation used the original `grasp_test` settings, which differ from the paper in ways that
matter (most importantly the object mass). Neither evaluation is directly comparable to the paper's numbers;
see [Caveats](#caveats).

## Summary

| Metric | H→A (Human→Allegro) | H→S (Human→Shadow) |
|---|---|---|
| Grasps evaluated | 1,016 | 1,009 |
| Mean IoU (GWH similarity) | 0.179 | 0.181 |
| Contact consistency | 0.00129 | 0.00102 |
| Mean std (joint-angle spread) | 0.307 | 0.202 |
| Isaac success rate, original settings | 436 / 1,016 = 42.9% | 241 / 1,009 = 23.9% |
| Isaac success rate, paper settings | **615 / 1,016 = 60.5%** | **398 / 1,009 = 39.4%** |

`sample.py` generates 32 samples per object (1,024 per direction); samples whose GWH hull could not be
computed are dropped, which leaves 1,016 and 1,009.

Mean std is not comparable across hands: Allegro has 16 joints and Shadow has 22, with different ranges.

## Isaac success rate by object

Original settings (`grasp_test_force.yaml`). Successful grasps / grasps evaluated. **Bold** marks objects where H→S did better than H→A.

| Object | H→A | H→S |
|---|---|---|
| 3D_Dollhouse_Happy_Brother | 28/32 | 12/32 |
| 3D_Dollhouse_Lamp | 23/32 | 20/31 |
| ACE_Coffee_Mug_Kristen_16_oz_cup | 31/32 | 13/32 |
| Android_Figure_Panda | 30/32 | 25/32 |
| Asus_80211ac_DualBand_Gigabit_Wireless_Router_RTAC68R | 0/32 | 0/30 |
| BIA_Cordon_Bleu_White_Porcelain_Utensil_Holder_900028 | 13/32 | **14/32** |
| Black_Forest_Fruit_Snacks_28_Pack_Grape | 2/31 | 0/31 |
| Clue_Board_Game_Classic_Edition | 0/31 | 0/32 |
| Cole_Hardware_Deep_Bowl_Good_Earth_1075 | 3/30 | **8/31** |
| Cole_Hardware_Mini_Honey_Dipper | 23/31 | 13/31 |
| Corningware_CW_by_Corningware_3qt_Oblong_Casserole_Dish_Blue | 9/32 | **12/29** |
| Design_Ideas_Drawer_Store_Organizer | 9/31 | 7/32 |
| Ecoforms_Garden_Pot_GP16ATurquois | 17/32 | 9/32 |
| Ecoforms_Pot_Nova_6_Turquoise | 13/32 | 13/32 |
| Grreat_Choice_Dog_Double_Dish_Plastic_Blue | 4/32 | **8/32** |
| Logitech_Ultimate_Ears_Boom_Wireless_Speaker_Night_Black | 12/32 | 0/32 |
| Nikon_1_AW1_w11275mm_Lens_Silver | 29/32 | 9/32 |
| Nintendo_2DS_Crimson_Red | 21/31 | 2/31 |
| Nordic_Ware_Original_Bundt_Pan | 6/32 | **20/32** |
| Ortho_Forward_Facing_CkAW6rL25xH | 0/32 | 0/32 |
| Razer_Abyssus_Ambidextrous_Gaming_Mouse | 26/32 | 12/32 |
| Razer_Blackwidow_Tournament_Edition_Keyboard | 8/32 | 3/32 |
| Razer_Goliathus_Control_Edition_Small_Soft_Gaming_Mouse_Mat | 16/32 | 6/32 |
| Remington_TStudio_Hair_Dryer | 7/32 | 2/32 |
| Sushi_Mat | 19/32 | 9/31 |
| TERREX_FAST_X_GTX | 9/32 | 1/32 |
| Threshold_Hand_Towel_Blue_Medallion_16_x_27 | 7/32 | 1/31 |
| Threshold_Porcelain_Pitcher_White | 2/32 | 0/32 |
| Threshold_Salad_Plate_Square_Rim_Porcelain | 15/32 | 8/31 |
| Timberland_Mens_Earthkeepers_Stormbuck_Lite_Plain_Toe_Oxford | 22/32 | 6/31 |
| W_Lou_z0dkC78niiZ | 7/32 | 1/31 |
| YumYum_D3_Liquid | 25/31 | 7/32 |
| **Total** | **436/1,016** | **241/1,009** |

Observations:
- IoU and contact consistency are nearly the same for both directions.
- Four of the five objects where H→S did better are concave (bowl, Bundt pan, dog dish, casserole dish).
  This lines up with the difference in object collision shapes between the two Isaac tasks (see below).
- The router, the board game and `Ortho_Forward_Facing` fail in both directions.

## Evaluation matched to the paper

The paper's Isaac test: push the object with a uniform acceleration of 0.5 m/s² along ±x, ±y, ±z for 60 steps
each; a grasp succeeds if the object moves less than 2 cm in all six directions; friction 10; object density
10000; Isaac Gym's standard position control. Settings the paper does not mention were left as in the
authors' code.

| Item | Paper | Original `grasp_test` | Paper settings (`grasp_test_force_paper.yaml`) |
|---|---|---|---|
| Steps per direction | 60 | 50 | 60 |
| Object mass | density 10000 | `mass=0.1` from the generated URDFs, which overrides the density | URDFs without mass (`make_object_urdfs.py --no_mass`), so Isaac Gym uses density 10000 |
| Push | uniform 0.5 m/s² | force `5000 × mesh volume` N | same force (authors' code) |
| Resulting acceleration (32 test objects) | 0.5 m/s² | 0.52–349 m/s² | 0.14–0.50 m/s² |
| Object collision | not stated | Allegro: single convex hull; Shadow: VHACD | VHACD for both |
| Success, friction, directions | as above | same | same |
| Not in the paper (unchanged) | | object damping (linear 10, angular 100), finger position control (Allegro: gains 1e10 with targets at the generated pose + 0.2 rad; Shadow: stiffness/damping 400 with the closure-optimized pose), physics parameters (dt 1/60, 2 substeps, 4 position iterations) | same |

The force stays `5000 × mesh volume`, which equals density × volume × 0.5 only if Isaac Gym's mass equals
density × mesh volume. Isaac Gym computes the mass from the collision shapes, which are larger than the mesh
for hollow objects, so the acceleration is below 0.5 m/s² for them (about 0.14–0.2 m/s² for the pots and the
Bundt pan). Per-object values are in `paper_eval/object_accel.csv`.

### Ground-truth grasps, step by step
The same 256 ground-truth grasps per hand (32 test objects × 8 random dataset grasps, from
`grasp_test/make_gt_samples.py`), with one change added at each stage:

| Stage | Settings | Allegro | Shadow |
|---|---|---|---|
| 0 | Original (`grasp_test_force.yaml`) | 95 / 256 = 37.1% | 41 / 256 = 16.0% |
| 1 | + URDFs without mass (density 10000) + 60 steps (`paper_eval/configs/stage1.yaml`) | 176 / 256 = 68.8% | 90 / 256 = 35.2% |
| 2 | + VHACD for both tasks (`grasp_test_force_paper.yaml`) | 134 / 256 = 52.3% | 87 / 256 = 34.0% |

- Stage 0 → 1: the mass fix removes pushes of up to 349 m/s², which failed most large objects.
- Stage 1 → 2: with VHACD, hollow objects get a smaller collision volume, hence a smaller mass and a larger
  acceleration (closer to 0.5 m/s²), and different contacts. The Shadow task already used VHACD.
- The ground-truth grasps are random dataset grasps, including ones that fail, so they are not an upper
  bound for the generated grasps.

### Generated grasps under the paper settings
Successful grasps / grasps evaluated, original settings → paper settings:

| Object | H→A original | H→A paper | H→S original | H→S paper |
|---|---|---|---|---|
| 3D_Dollhouse_Happy_Brother | 28/32 | 22/32 | 12/32 | 12/32 |
| 3D_Dollhouse_Lamp | 23/32 | 25/32 | 20/31 | 21/31 |
| ACE_Coffee_Mug_Kristen_16_oz_cup | 31/32 | 24/32 | 13/32 | 18/32 |
| Android_Figure_Panda | 30/32 | 28/32 | 25/32 | 25/32 |
| Asus_80211ac_DualBand_Gigabit_Wireless_Router_RTAC68R | 0/32 | 2/32 | 0/30 | 0/30 |
| BIA_Cordon_Bleu_White_Porcelain_Utensil_Holder_900028 | 13/32 | 24/32 | 14/32 | 19/32 |
| Black_Forest_Fruit_Snacks_28_Pack_Grape | 2/31 | 5/31 | 0/31 | 0/31 |
| Clue_Board_Game_Classic_Edition | 0/31 | 2/31 | 0/32 | 0/32 |
| Cole_Hardware_Deep_Bowl_Good_Earth_1075 | 3/30 | 18/30 | 8/31 | 22/31 |
| Cole_Hardware_Mini_Honey_Dipper | 23/31 | 19/31 | 13/31 | 13/31 |
| Corningware_CW_by_Corningware_3qt_Oblong_Casserole_Dish_Blue | 9/32 | 25/32 | 12/29 | 15/29 |
| Design_Ideas_Drawer_Store_Organizer | 9/31 | 25/31 | 7/32 | 11/32 |
| Ecoforms_Garden_Pot_GP16ATurquois | 17/32 | 25/32 | 9/32 | 32/32 |
| Ecoforms_Pot_Nova_6_Turquoise | 13/32 | 21/32 | 13/32 | 32/32 |
| Grreat_Choice_Dog_Double_Dish_Plastic_Blue | 4/32 | 23/32 | 8/32 | 15/32 |
| Logitech_Ultimate_Ears_Boom_Wireless_Speaker_Night_Black | 12/32 | 22/32 | 0/32 | 0/32 |
| Nikon_1_AW1_w11275mm_Lens_Silver | 29/32 | 31/32 | 9/32 | 19/32 |
| Nintendo_2DS_Crimson_Red | 21/31 | 22/31 | 2/31 | 6/31 |
| Nordic_Ware_Original_Bundt_Pan | 6/32 | 23/32 | 20/32 | 31/32 |
| Ortho_Forward_Facing_CkAW6rL25xH | 0/32 | 7/32 | 0/32 | 3/32 |
| Razer_Abyssus_Ambidextrous_Gaming_Mouse | 26/32 | 18/32 | 12/32 | 15/32 |
| Razer_Blackwidow_Tournament_Edition_Keyboard | 8/32 | 13/32 | 3/32 | 3/32 |
| Razer_Goliathus_Control_Edition_Small_Soft_Gaming_Mouse_Mat | 16/32 | 25/32 | 6/32 | 11/32 |
| Remington_TStudio_Hair_Dryer | 7/32 | 12/32 | 2/32 | 7/32 |
| Sushi_Mat | 19/32 | 25/32 | 9/31 | 11/31 |
| TERREX_FAST_X_GTX | 9/32 | 22/32 | 1/32 | 8/32 |
| Threshold_Hand_Towel_Blue_Medallion_16_x_27 | 7/32 | 8/32 | 1/31 | 2/31 |
| Threshold_Porcelain_Pitcher_White | 2/32 | 10/32 | 0/32 | 9/32 |
| Threshold_Salad_Plate_Square_Rim_Porcelain | 15/32 | 27/32 | 8/31 | 15/31 |
| Timberland_Mens_Earthkeepers_Stormbuck_Lite_Plain_Toe_Oxford | 22/32 | 23/32 | 6/31 | 12/31 |
| W_Lou_z0dkC78niiZ | 7/32 | 13/32 | 1/31 | 5/31 |
| YumYum_D3_Liquid | 25/31 | 26/31 | 7/32 | 6/32 |
| **Total** | **436/1,016** | **615/1,016** | **241/1,009** | **398/1,009** |

Hollow or heavy objects (bowl, pots, Bundt pan, dog dish, casserole dish) gain the most. A few objects drop
even though their push became much weaker (e.g. the mug for H→A: 31 → 24, acceleration 9.7 → 0.35 m/s²;
the Razer mouse: 26 → 18, 9.2 → 0.45 m/s²). The cause was not isolated; the collision shape (convex hull →
VHACD for the Allegro task) and the longer push (50 → 60 steps) also changed.

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
| Test objects | 32, from the `test` split in `data/grasp_data/HumanHand/split.json` |
| Samples | 32 per object, SDE integrated with Euler, `dt=0.005` (`sample.py` defaults), seed 42 |
| IoU | Monte Carlo, 500,000 samples (`eval_samples.py`) |
| Isaac stability test | Original: `grasp_test/envs/tasks/grasp_test_force.yaml`. Paper settings: `grasp_test/envs/tasks/grasp_test_force_paper.yaml` (see above) |

The setup steps are in the README section "Setup on RTX 50-series (Blackwell) GPUs".

### Ground-truth sanity checks
Ground-truth dataset grasps run through the same Isaac tasks (`grasp_test/make_gt_samples.py`).
Both runs are very small:

| Hand | Objects × grasps | Success |
|---|---|---|
| Allegro | 3 × 16 | 22 / 48 = 45.8% |
| Shadow | 2 × 8 | 3 / 16 = 18.8% |

## Caveats
- **Split.** `HumanHand/split.json` and `shadow_hand/split.json` were derived from a locally generated
  `Allegro/split.json`. If that differs from the split the checkpoints were trained with, some "test"
  objects may have been seen during training, which would inflate the scores.
- **The two Isaac tasks are not equivalent**, so the success-rate gap does not directly measure model quality:
  - Object collision: the Allegro task uses a single convex hull of the object; the Shadow task
    convex-decomposes the object with VHACD (e.g. 64 hulls for `Ecoforms_Garden_Pot`).
  - The ground-truth sanity checks above suggest the Shadow task is harder even with dataset grasps.
- **Object URDFs** are generated by `grasp_test/make_object_urdfs.py` (the repository does not ship them):
  one link with the `.obj` as visual and collision mesh. By default the link has `mass=0.1`, which Isaac Gym
  keeps; this made the original evaluation push objects far harder than the paper. The paper settings use
  URDFs without mass (`--no_mass`).
- **The Shadow task is not deterministic.** Re-running the same grasps can change individual results
  (e.g. 1/4, 2/4 and 3/4 on the same four mug grasps), so Shadow success rates carry a few points of noise.
  The Allegro task reproduced the same results in the one repeat that was tried.
- **CPU torch pipeline.** The Isaac tasks were patched to run torch on CPU (`--device cpu`), because no
  PyTorch build for Python 3.8 supports sm_120. PhysX still runs on the GPU. The authors used the GPU pipeline.
- **Library versions** differ from `environment.yml` (PyTorch 2.7.1 instead of 1.12.1, pytorch3d 0.7.9,
  xformers 0.0.31), and some libraries were pinned or patched to keep the code working; see the README.
- **Data loader fix.** Per-grasp files are named by the original grasp index, which skips grasps dropped in
  preprocessing. The loader now maps metadata row k to the k-th file in sorted order (`dataset/data_utils.py`).
- **Preprocessing** was run only for the 32 test objects (human and Shadow hands); Allegro data was
  preprocessed beforehand.

## Files

| Path | Contents |
|---|---|
| `isaac_h2a/succ.pickle`, `isaac_h2s/succ.pickle` | Per-object list of per-grasp success flags (`{object_id: [bool, ...]}`) |
| `isaac_h2a/evaluation_right.log`, `isaac_h2s/evaluation_right.log` | Per-object success rates and summary |
| `isaac_h2a/run.log`, `isaac_h2s/run.log` | Full stdout/stderr of the Isaac runs |
| `paper_eval/stage{0,1,2}_*/gt_{allegro,shadow}/` | Ground-truth runs per stage (`succ.pickle`, `evaluation_right.log`) |
| `paper_eval/stage2_paper/isaac_{h2a,h2s}/` | Generated grasps under the paper settings (`succ.pickle`, `evaluation_right.log`, `run.log`) |
| `paper_eval/object_accel.csv` | Per-object mesh volume, push force, and Isaac Gym mass / acceleration for each setting |
| `paper_eval/configs/stage1.yaml` | Config of stage 1 (the stage 2 config is `grasp_test/envs/tasks/grasp_test_force_paper.yaml`) |

Not included because of size: the full `samples.pkl` files (about 10 GB for H→A and 7.7 GB for H→S,
in `logs/diffusion_ddp/mgg/<method>/test_output/`), which hold the generated grasps, hulls, point clouds
and per-sample IoU.
