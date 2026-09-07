# How to Run HybridPointTransformer

A step-by-step guide to reproduce the 53-class dental segmentation experiments.

---

## 1. What This Project Does

- Segments 3D intraoral scans into **53 classes** (FDI tooth numbering + gingiva/background)
- Uses a hybrid architecture: **PointNet++ encoder + Transformer attention + PointNet++ decoder**
- Input: `.obj` files (6-D vertices: XYZ ) + `.json` files (per-vertex FDI labels)
- Output: per-vertex class predictions across 65,536 sampled points

---

## 2. Dataset

### 2.1 Source
- **3DTeethSeg'22** challenge dataset (MICCAI 2022)
- **1,800 intraoral scans from 900 patients** (2 scans per patient: upper + lower jaw)
- Scanners: Primescan, Trios3, iTero Element
- 89,463–300,000 points per raw scan

### 2.2 Classes (53 total)

| Category | FDI Codes | Count |
|---|---|---|
| Gingiva / Background | 0 | 1 |
| Permanent teeth (Upper Right) | 11–18 | 8 |
| Permanent teeth (Upper Left) | 21–28 | 8 |
| Permanent teeth (Lower Left) | 31–38 | 8 |
| Permanent teeth (Lower Right) | 41–48 | 8 |
| Primary teeth (Upper Right) | 51–55 | 5 |
| Primary teeth (Upper Left) | 61–65 | 5 |
| Primary teeth (Lower Left) | 71–75 | 5 |
| Primary teeth (Lower Right) | 81–85 | 5 |

Class distribution is highly imbalanced: gingiva/background is ~65% of all points.

### 2.3 Data Split (patient-level, 70/20/10)
| Split | Patients | Scans |
|---|---|---|
| Training | 630 (70%) | 1,260 |
| Validation | 180 (20%) | 360 |
| Test | 90 (10%) | 180 |

Both scans of the same patient always go to the same split (no data leakage).

---

## 3. Prerequisites

Install the following:
- Python >= 3.8
- PyTorch >= 1.13
- torch-geometric
- torch-points3d
- open3d, trimesh, scikit-learn, numpy, tqdm, pandas, einops, optuna, rarfile, vedo

---

## 4. Step-by-Step Setup

### Step 1: Download the Dataset
Download the 3DTeethSeg'22 dataset from the [MICCAI 2022 challenge repository](https://github.com/abenhamadou/3DTeethSeg_MICCAI_Challenges).

### Step 2: Organize the Data
Run the data splitting script to create the `Dataset_900/` directory with the 70/20/10 train/validation/test split:

```
Dataset_900/
├── train/
│   ├── data/          ← .obj files
│   └── ground-truth/  ← .json files
├── validation/
│   ├── data/
│   └── ground-truth/
├── test/
│   ├── data/
│   └── ground-truth/
├── train.txt
├── validation.txt
└── test.txt
```

Each `.obj` file contains 6-D vertex features (XYZ + RGB). Each `.json` file contains `{"labels": [...]}` with per-vertex FDI tooth codes.

### Step 3: Train the Model
Run the training script. Default hyperparameters:

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Focal gamma | 2.8214 |
| Epochs | 250 |
| Batch size | 1 |
| num_points | 65,536 |
| num_classes | 53 |
| d_model | 256 |
| nhead | 4 |
| Early stopping patience | 20 |

During training:
- Best model is saved when validation mIoU improves (`checkpoint_best.pth`)
- Backup checkpoints saved every 5 epochs

### Step 4: Test the Model
Run the test script to evaluate the trained checkpoint on the held-out test set. Results are saved .

### Step 5 (Optional): Hyperparameter Search
Run the training script with the `--optuna_trials` flag set to 20–50 to search for optimal hyperparameters via TPE sampler.

---

## 5. Preprocessing Pipeline

All point-based models share the same preprocessing (implemented in the dataset loader):

```
INPUT:  .obj file (N x 6: XYZ + RGB) + .json file (per-vertex FDI labels)
OUTPUT: pos (65,536 x 3), x (65,536 x 3), y (65,536,) 

1. READ
   - Parse vertices from .obj → features (N, 6) float32
   - Parse labels from .json → labels (N,) int64
   - Truncate to min(len(features), len(labels))

2. VERTEX CAP (training only)
   - If len(features) > 100,000: random subset to 100,000

3. AUGMENTATION (training only; skip for evaluation)
   a) Random Z-rotation: angle ~ Uniform(-pi/18, +pi/18) [±10°]
   b) Random scaling: scale ~ Uniform(0.9, 1.1) per axis
   c) Color jitter: RGB += Uniform(-0.1, +0.1), clip to [0, 1]
   d) Elastic distortion: granularity=[0.2, 0.8], magnitude=[0.4, 1.6]

4. NORMALIZATION
   - Center: pos ← pos − mean(pos, axis=0)
   - Unit sphere: pos ← pos / max(||pos||_2)

5. LABEL REMAPPING (FDI → [0, 52])
   - Background (0) → 0
   - Permanent teeth (11–18, 21–28, 31–38, 41–48) → 1–8, 9–16, 17–24, 25–32
   - Primary teeth (51–55, 61–65, 71–75, 81–85) → 33–37, 38–42, 43–47, 48–52

6. FIXED-POINT SAMPLING → 65,536 points
   - If N < 65,536: sample with replacement
   - If N >= 65,536: sample without replacement
```

Region removal (optional preprocessing step): Triangle Density Analysis via `vedo` removes gingival tissue and alveolar bone below tooth crowns in lower jaw scans before training.

---

## 6. Model Architecture

```
Input: pos (B, 65536, 3) + x (B, 65536, 3)
Concat → (B, 65536, 6)

ENCODER (PointNet++ Set Abstraction)
├── SA1: FPS→1,024, r=0.1, K=32, MLP(64,64,128)    → (B, 1024, 128)
├── SA2: FPS→256, r=0.2, K=32, MLP(128,128,256)     → (B, 256, 256)
└── SA3: FPS→64, r=0.4, K=32, MLP(256,256,256)      → (B, 64, 256)

TRANSFORMER (3 blocks)
├── Multi-head self-attention: d_model=256, nhead=4, dropout=0.5
├── Feed-forward: GELU(MLP(256→512→256))
└── LayerNorm + residual (post-norm)

DECODER (PointNet Feature Propagation)
├── FP3: 64→256 pts, fuse SA2, MLP(256,256)
├── FP2: 256→1,024 pts, fuse SA1, MLP(256,128)
└── FP1: 1,024→65,536 pts, fuse raw XYZ, MLP(128,128,128)

SEGMENTATION HEAD
├── Conv1d(128→128) + BatchNorm + ReLU + Dropout(0.5)
└── Conv1d(128→53) → raw logits (B, 65536, 53)
```

| Component | Input Dim | Output Dim |
|---|---|---|
| SA1 | 9 (3xyz + 6 input) | 128 |
| SA2 | 131 (128feat + 3xyz) | 256 |
| SA3 | 259 (256feat + 3xyz) | 256 |
| Transformer | 256 | 256 |
| FP3 | 512 (256+256 skip) | 256 |
| FP2 | 256 (128+128 skip) | 128 |
| FP1 | 131 (128+3 skip) | 128 |
| SegHead | 128 | 53 |

---

## 7. Loss Function

Class-weighted focal loss:

```
L_focal = Σ_c −(1 − p_c)^γ · log(p_c) · α_c
```

- `p_c` = softmax probability for class `c`
- `γ` (focal_gamma) = 2.8214 (Optuna-optimized)
- `α_c` = class weight computed as: `w_c = 1 / log(1.02 + |C_c|)`, then normalized: `w_c ← w_c / (Σ_c w_c) × 53`
- `|C_c|` = number of training points belonging to class `c`
- Zero-count classes smoothed to 1 to avoid `log(0)`

Implemented as `FocalLoss` class in the training script. Uses `F.cross_entropy(..., reduction='none')` internally, then applies focal modulation and class weighting.

---

## 8. Hyperparameter Optimization

If you want to search for better hyperparameters (optional):

| Parameter | Search Range | Scale |
|---|---|---|
| Learning rate | [1e-6, 1e-2] | Log |
| Batch size | 1 | Fixed |
| Optimizer | {Adam, AdamW} | Categorical |
| Weight decay | [1e-6, 1e-3] | Log |
| Focal gamma | [1.0, 3.0] | Linear |

- Sampler: TPE (Tree-structured Parzen Estimator)
- Trials: 20–50 per architecture
- Objective: maximize validation mIoU

Best found configuration: AdamW, lr=1e-4, weight_decay=1e-4, focal_gamma=2.8214.

---

## 9. Other Models

The same dataset, split, and preprocessing pipeline are used for all models. The repo includes training and evaluation scripts for 8 baseline architectures alongside the proposed HybridPointTransformer:

- **PointNet** — vanilla point cloud network
- **PointNet++** — hierarchical point cloud network with set abstraction
- **Transformer** — vanilla transformer on point clouds
- **Superpoint Transformer (SPT)** — superpoint-based transformer
- **TeethGNN** — graph neural network for dental segmentation
- **KPConv** — kernel point convolution
- **DentalMAN** — dental-specific attention network
- **Custom DGCNN (Dilated)** — dilated dynamic graph CNN

To train any of these, run its corresponding training script (e.g., `python Train_PointNetPP.py`). To evaluate, run its test script (e.g., `python Test_PointNetPP.py`). The workflow is identical — only the model architecture and dataloader differ.

---

## 10. Code Structure

| File | Purpose |
|---|---|
| `Train_Hybrid.py` | Training loop, FocalLoss, checkpointing, early stopping |
| `Test_Hybrid.py` | Model evaluation on test set, per-class metrics |
| `networks/HybridTransformer.py` | HybridPointTransformer + TransformerBlock |
| `networks/HybridTransformerPointnet2_utils.py` | PointNetSetAbstraction, PointNetFeaturePropagation |
| `dataloader/DataLoader_HybridTransformer.py` | ObjDataset, data augmentation, train/val/test splitting |
| `dataloader/PreprocessingPaper.py` | Shared preprocessing: normalization, label remapping, class weights |
| `Split_Data_900.py` | Patient-level 70/20/10 data split |

Each baseline model follows the same pattern — its own `Train_<Model>.py`, `Test_<Model>.py`, network file under `networks/`, and dataloader under `dataloader/`.

---

