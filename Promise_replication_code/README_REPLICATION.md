# Replication of SUGFW+ on Promise12 (prostate MRI)

This document describes our replication of **SUGFW+** (Ma et al., *An Uncertainty-guided Feature Weighting Framework for Cold Start Active Adaptation of SAM in Medical Image Segmentation*, arXiv:2608.16110) using the code released by the authors. It covers the replication of the **Promise12** dataset only. Every table below is followed by the exact
commands that produce it. The authors' own README is in [`README.md`](README.md).
All paths below are relative to this folder (`Promise_replication_code/` in the group repository); `data/`,
`checkpoint/` and `output/` are created locally and are not committed.

## 1. Summary

| | |
|---|---|
| **What we replicated** | The main result for Promise12 (annotation budgets 3% to 8%) and three ablations at 3% and 4% (w/o PGDR, w/o GSCU, α_max = 0.2). |
| **Main finding** | The trend is reproduced: Dice rises with the budget. Stable runs sit about 3 to 4 Dice points below the paper (e.g. 83.0 vs 86.4 at 3%, 86.2 vs 89.3 at 8%). At 5%, our two stable seeds (87.0 and 87.6) match the paper (87.9). |
| **Instability** | 2 of 18 training runs collapsed (5% seed 1234, 7% seed 1233).|
| **Ablations** | With 2 seeds, none of the variants differs from the full method by more than the seed-to-seed spread (about 1 to 3 Dice points), so we couldn't reproduce the paper's ablation gains. |
| **Not reported** | HD95. The released code computes it in voxels, not millimetres. |

## 2. What we changed in the authors' code

| File | Change | Why |
|---|---|---|
| `MA-SAM-un/inference.py` | Resize the 512×512 prediction back to the label size before computing the metrics. | The shipped code crashed on a 512 vs 256 shape mismatch (the resize lines were commented out). |
| `UC-SAM/command/get_un_Promise12.sh` | Point `--csv_path` at Promise12 (it pointed at UTAH). | Wrong dataset. |
| `UC-SAM/get_avg_un.py` | Promise12 paths, and a 256×256 array (was Liver paths and 512×512). | Hard-coded for another dataset. |
| `UC-SAM/select_samples.py` | `__main__` block set to Promise12, ratios 3 to 8%, SUGFW+ selection only. | Baselines were not part of this replication. |
| `MA-SAM-un/command/Promise12/run_budgets.sh` (new) | Trains and tests every budget and seed. | The original `run_my.sh` loops over 120 configurations. |
| `UC-SAM/make_ablation_lists.py` (new) | Creates the query lists for the ablations. The w/o GSCU selection is seeded. | The original random choice inside each cluster was unseeded. |
| `MA-SAM-un/command/Promise12/run_ablation.sh` (new) | Trains and tests the 12 ablation runs. | |

Training code, model code, hyper-parameters and the selection algorithms are unchanged
(SAM ViT-B, LoRA rank 32, AdamW, lr 0.0008, 300 epochs, batch size 16, λ = 6 i.e. `soft_scale=6`,
K = 10 augmentations, `--mse_param 0.02`).

## 3. Environment

Hardware used: **UCloud (DeiC Interactive HPC)**, one full NVIDIA B200, 48 vCPUs (AMD EPYC 9655), 288 GB RAM.
Software: Ubuntu 24.04, Python 3.12, PyTorch 2.11 (CUDA 12.8). The authors report four RTX 2080 Ti GPUs; we use
a single GPU with the same total batch size of 16.

The pinned versions in `pyproject.toml` (2023) have no Python 3.12 builds, so we did not use `pip install -e .`.
Only these packages are imported by the code:

```sh
# torch with CUDA 12.8 (needed for Blackwell GPUs such as the B200); skip if your image already has it
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install "setuptools<82" "numpy<2" scipy pandas tqdm icecream SimpleITK pillow \
    scikit-learn scikit-image medpy einops opencv-python-headless \
    pycocotools nibabel matplotlib imageio

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
```
On the B200 this prints `... True (10, 0)`. The code uses `.cuda()` throughout, so a CUDA GPU is required. 

SAM checkpoint:
```sh
mkdir -p checkpoint && cd checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
cd ..
```

## 4. Data and preprocessing

Download Promise12 (https://zenodo.org/records/8026660) and arrange it as
`data/Promise12/raw_data/{training_data,livechallenge_test_data,test_data}`, each containing the `.mhd/.raw`
image files and the matching `*_segmentation.mhd/.raw` files. 

```sh
cd preprocessing
python Promise12_pre.py     # per-slice .pkl files in data/Promise12/Foreslices/{image,label}
python Promise12_csv.py     # splits/{train,valid,test}.csv
cd ..
```

**Dataset.** 100 prostate T2-weighted MRI volumes. 

**Split** (by case, following the official folders, 5:2:3):

| Set | Source folder | Cases | Slices |
|---|---|---|---|
| Train (query pool) | `training_data` | 0 to 49 | 960 |
| Validation | `livechallenge_test_data` | 50 to 69 | 353 |
| Test | `test_data` | 70 to 99 | 529 |

Annotation budgets are counted in slices of the 960-slice training pool (3% = 28 slices, 4% = 38, 5% = 48,
6% = 57, 7% = 67, 8% = 76).

**Preprocessing** (authors' code): clip intensities above the 99th percentile (99.5 in the paper); z-score normalise each volume; resize
each slice in-plane to 256×256 (cubic for images, nearest for labels; the slice direction is untouched);
**drop slices more than 2 positions away from the first or last slice containing prostate**
(2,726 → 1,842 slices). At training and test time each slice is min-max normalised, copied to three channels and
upsampled to 512×512 for SAM; predictions are resized back to 256×256 for evaluation.

**3D handling.** As described in the paper and the code, the model works in 2D: slices are selected, annotated, and segmented independently. For evaluation, the predictions are stacked per case, and the Dice Similarity Coefficient (DSC) and the 95th percentile Hausdorff Distance (HD95) are computed in 3D and averaged over the test cases of the respective dataset.

## 5. Running the pipeline

All commands are run from the directory shown. Keep the folder layout above, because paths are relative.

### 5.1 Uncertainty and features (about 80 minutes on one B200; the slowest step)
```sh
cd UC-SAM
bash command/get_un_Promise12.sh        # writes feature/ and uncertainty/ next to splits/
python get_avg_un.py                    # writes uncertainty/uncertainty_10_avg.pkl
```
Run it inside `tmux`; the script resumes if interrupted. 

### 5.2 Query sets for all budgets
```sh
python select_samples.py                # SUGFW+ (soft_scale=6, un=10) for 3-8%
ls ../data/Promise12/Foreslices/splits | grep my_select_
# my_select_{28,38,48,57,67,76}_scale6_un10.csv
```
The selection is deterministic (`KMeans(random_state=42)`, no random choice in the greedy step).

### 5.3 Main experiment: all budgets, 3 seeds (about 3 hours per seed)
```sh
cd ../MA-SAM-un
bash command/Promise12/run_budgets.sh
```
For each budget (28, 38, 48, 57, 67, 76) and seed (1234, 1233, 1235) it trains for 300 epochs, tests the best checkpoint. A single run by hand:
```sh
python train.py -b 16 --n_gpu 1 \
  --root_path ../data/Promise12/Foreslices \
  --csv_path ../data/Promise12/Foreslices/splits/my_select_28_scale6_un10.csv \
  -o ../output/Promise12/sugfw_28_seed1234 --seed 1234 --train_max_epochs 300

python inference.py --adapt_ckpt ../output/Promise12/sugfw_28_seed1234/train_best_model.pth \
  --root_path ../data/Promise12/Foreslices --stage test \
  --num_classes 2 --img_size 512 --seed 1234 --cal_hd95
```
Outputs: `MA-SAM-un/training_log/<run>_log.txt` (losses and validation Dice at epochs 100, 200 and 300) and
`MA-SAM-un/testing_log/Promise12/<run>__log.txt` (per-case Dice, and the mean ± std in the last line).
The checkpoint of the best validation epoch is `output/Promise12/<run>/train_best_model.pth`.

### 5.4 Ablations (12 runs, about 5 hours)
```sh
cd ../UC-SAM
python make_ablation_lists.py           # w/o PGDR and w/o GSCU query lists for 3% and 4%
cd ../MA-SAM-un
bash command/Promise12/run_ablation.sh
```
`run_ablation.sh` trains, for budgets 28 and 38 and seeds 1234 and 1233: `abl_alpha02` (full method lists, `--mse_param 0.2`),
`abl_pgdr` (`my_select_{n}_wo_featureweighting.csv`) and `abl_gscu` (`my_select_{n}_wo_GS_seed{seed}.csv`). It prints
a summary that lists each ablation next to the corresponding full-method run (`sugfw_{n}_seed{seed}`), so run
5.3 first.

### 5.5 Reading the results 

Dice in the logs is a fraction (0.8174 = 81.74%).

## 6. Results

All numbers are the Dice similarity coefficient (%) on the 30 test cases (cases 70 to 99), averaged over cases, for the
checkpoint with the best validation Dice. The paper's numbers are SUGFW+ on Promise12.

### 6.1 Main result (produced by 5.1 to 5.3)

| Budget | Slices | Seed 1234 | Seed 1233 | Seed 1235 | Paper |
|---|---|---|---|---|---|---|
| 3% | 28 | 81.74 | 84.13 | 83.14 | 86.38 |
| 4% | 38 | 84.21 | 83.15 | 81.37 | 87.35 |
| 5% | 48 | **47.13** | 87.02 | 87.55 | 87.86 |
| 6% | 57 | 84.73 | 83.66 | 84.87 | 88.51 |
| 7% | 67 | 87.04 | **56.11** | 85.70 | 89.20 |
| 8% | 76 | 86.07 | 85.95 | 86.47 | 89.25 |

The std is over seeds. The per-case std within a run is typically 4 to 8 Dice points. Bold values are failed runs
(see below). We do not drop them.

**Failed runs.** In the two bold runs, validation Dice fell or oscillated during training (5%, seed 1234, seed 1233), and the training loss stayed high (at 5%/seed 1234 the Dice loss was
0.19 at the end, versus 0.009 for the healthy 4% run). This is an optimisation failure, not a problem with the query
set (the 5% set looks like the others: 31 different cases, normal foreground fraction). Other seeds with the same
query set train normally. We did not change the learning rate to try to avoid it.

### 6.2 Ablation (produced by 5.4), seeds 1234 and 1233

| Variant | 3%: seed 1234 | 3%: seed 1233 | 4%: seed 1234 | 4%: seed 1233 |
|---|---|---|---|---|---|---|
| Full method (from 6.1) | 81.74 | 84.13 | 84.21 | 83.15 |
| α_max = 0.2 | 81.88 | 84.11 | 84.43 | 83.21 |
| w/o PGDR | 84.59 | 81.34 | 83.59 | 84.51 |
| w/o GSCU | 82.60 | 81.66 | 86.13 | 83.08 |

Paper, Table 5 (Promise12; same pipeline, selection ablations only):

| | 3% | 4% |
|---|---|---|
| SUGFW (full) | 86.38 | 87.35 |
| w/o PGDR | 85.13 | 85.30 |
| w/o GSCU | 81.22 | 85.32 |
| Random selection | 84.03 | 85.72 |





