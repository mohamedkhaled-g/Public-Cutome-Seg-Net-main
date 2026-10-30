# HybridPointTransformer: Multi-class Segmentation on Digitalized 3D Dental Models

Official implementation of the paper **"HybridPointTransformer: Multi-class Segmentation on Digitalized 3D Dental Models"**.

This repository contains the training and evaluation code for a hybrid point-cloud architecture that combines PointNet++ style local feature aggregation with Transformer-based global attention for 53-class FDI-compliant dental segmentation (32 permanent teeth, 20 primary teeth, and the gingiva/background).

The model processes 65,536-point 6-D point clouds (XYZ + RGB) and is trained with a class-weighted focal loss tuned via Optuna. Reported results on the 3DTeethSeg'22 dataset (1,800 intraoral scans from 900 patients, 70/20/10 split): **90.77% validation accuracy / 87.90% mIoU** and **90.44% test accuracy / 87.77% mIoU**, outperforming PointNet, PointNet++, DGCNN, GNN, and KPConv baselines.

## Paper

- Title: *HybridPointTransformer: Multi-class Segmentation on Digitalized 3D Dental Models*
- Method: PointNet++ set abstraction encoder (1024 -> 256 -> 64 points) + 3 Transformer blocks (d_model=256, 4 heads) + PointNet feature-propagation decoder with skip connections
- Loss: class-weighted focal loss, `w_c = (1 / log(1.02 + n_c))` normalized, `gamma = 2.8214`
- Optimizer: AdamW (`lr = 1e-4`, `weight_decay = 1e-4`) or per-model Table 2 hyperparameters found by Optuna (50 trials, TPE)

## Results

### Final HybridPointTransformer model (65,536 points, epoch 234)

| Split | Accuracy | mIoU |
|---|---:|---:|
| Train | 91.73% | 89.24% |
| Validation | 90.77% | 87.90% |
| Test | 90.44% | 87.77% |

### Effect of the number of input points

Validation performance of the HybridPointTransformer (learning rate 1e-4, batch size 1) at selected epochs. Bold marks the lowest validation loss and the highest accuracy/mIoU of each run.

**30,000 input points** — best accuracy (0.8686) and mIoU (0.8280) at epoch 149.

| Epochs | Learning Rate | Points | Val. Loss | Val. Acc. | Val. mIoU |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.00010 | 30,000 | **0.0347** | 0.8412 | 0.7814 |
| 100 | 0.00010 | 30,000 | 0.0829 | 0.7588 | 0.6069 |
| 149 | 0.00010 | 30,000 | 0.0414 | **0.8686** | **0.8280** |
| 250 | 0.00010 | 30,000 | 0.0626 | 0.8573 | 0.8034 |

**65,536 input points** — best overall performance at epoch 234 (accuracy 0.9077, mIoU 0.8790); this checkpoint was selected as the final model.

| Epochs | Learning Rate | Points | Val. Loss | Val. Acc. | Val. mIoU |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.00010 | 65,536 | **0.0503** | 0.8610 | 0.7925 |
| 100 | 0.00010 | 65,536 | 0.0534 | 0.8708 | 0.8203 |
| 234 | 0.00010 | 65,536 | 0.0577 | **0.9077** | **0.8790** |
| 250 | 0.00010 | 65,536 | 0.0765 | 0.8896 | 0.8689 |

The full 250-epoch training logs for both runs are provided in the [`logResults/`](logResults/) folder:

- [`logResults/training_history_30_1.csv`](logResults/training_history_30_1.csv) — 30,000 input points
- [`logResults/training_history_65_1.csv`](logResults/training_history_65_1.csv) — 65,536 input points (final model)

Each row is one epoch with the columns `epoch, lr, train_loss, train_accuracy, train_miou, val_loss, val_accuracy, val_miou, val_tooth_accuracy`.

Increasing the input from 30,000 to 65,536 points raised the best validation accuracy from 0.8686 to 0.9077 (+3.9 pts) and mIoU from 0.8280 to 0.8790 (+5.1 pts), so 65,536 points is used for the final model.

### Comparison with baselines

| Model | Loss | Optimizer | LR | Weight Decay | Batch | Val. Acc. | Val. mIoU | Test Acc. | Test mIoU |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| HybridPointTransformer (Ours) | Focal | AdamW | 1e-4 | 1e-4 | 1 | 0.9077 | 0.8790 | 0.9044 | 0.8777 |
| PointNet | Focal | AdamW | 2.4e-5 | 2.1e-5 | 1 | 0.7000 | 0.4500 | 0.6083 | 0.3630 |
| PointNet++ | Focal | AdamW | 5.38e-4 | 1e-6 | 1 | 0.8000 | 0.5800 | 0.7366 | 0.4944 |
| DGCNN | Focal | AdamW | 1.1e-5 | 4.2e-5 | 1 | 0.7100 | 0.4900 | 0.6400 | 0.4100 |
| GNN (TeethGNN) | Cross-Entropy | Adam | 9.6e-5 | 8e-6 | 1 | 0.7800 | 0.5200 | 0.7125 | 0.4762 |
| KPConv | Cross-Entropy | AdamW | 2.8e-5 | 7.49e-4 | 1 | 0.6500 | 0.4200 | 0.6000 | 0.3800 |

## Requirements

- Python >= 3.8
- PyTorch >= 1.13
- torch-geometric
- torch-points3d (for `ElasticDistortion`)
- open3d, trimesh, scikit-learn (graph preprocessing)
- numpy, tqdm, pandas, einops
- optuna (hyperparameter optimization)

## Dataset

The public **3DTeethSeg'22** challenge dataset (MICCAI 2022) is used. Each sample consists of:

- `data/<case>/<case>_upper.obj` / `<case>_lower.obj` — 6-D vertex clouds (XYZ + RGB)
- `ground-truth/<case>/<case>_upper.json` / `<case>_lower.json` — per-vertex FDI labels

Prepare the 70/20/10 train/validation/test split at the patient level:

```
python Split_Data_900.py
```

This creates `Dataset_900/{train,validation,test}/{data,ground-truth}` plus per-scan mapping lists `train.txt`, `validation.txt`, `test.txt`. The final test scripts evaluate on `Dataset_900/test`.

## Repository Structure

```
dataloader/
  PreprocessingPaper.py          # shared Algorithm-1 preprocessing for all point-based models
  DataLoader_PNet.py             # PointNet
  DataLoader_PP.py               # PointNet++
  DataLoader_Cust_DGCNN.py       # DGCNN
  DataLoader_kpconv.py           # KPConv
  DataLoader_HybridTransformer.py# HybridPointTransformer (reference pipeline)
  Dataloader_TeethGnn.py         # raw facet-graph construction (10k facets)
  Dataloader_TeethGnn_Cached.py  # cached .pt graph dataset
networks/
  HybridTransformer.py           # proposed model
  PointNet.py, PointNetPlusPlus.py, Cust_DGCNN_Dilated.py, kpconv.py,
  TeethGnn_Model.py, TransformerNet.py, SuperpointTransformer.py, DentalMAN.py
Train_*.py                       # training + Optuna search for each model
Test_Hybrid.py, Test_PointNet.py, Test_PointNetPP.py, Test_Cust_DGCNN.py,
Test_kpconv.py, Test_TeethGnn.py # evaluation on the held-out test split
TeethGnnPreprocess_dataset.py    # build the TeethGNN .pt graph cache
Split_Data_900.py                # patient-level 70/20/10 split
logResults/                      # per-epoch training logs (CSV) for the 30,000- and 65,536-point runs
```

## Preprocessing (Algorithm 1)

All point-based models share `PreprocessingPaper.py`:

1. Read 6-D vertex features (XYZ + RGB) and ground-truth labels.
2. Training only: random Z-rotation within +/-10 deg, scaling in [0.9, 1.1], color jitter, and elastic distortion.
3. Centre on the centroid and normalise to unit radius.
4. Remap FDI codes to [0, 52] (53 classes).
5. Sample each scan to 65,536 points (uniform sampling after a 100k-vertex cap).

Class weights `w_c = (1 / log(1.02 + n_c)) / sum_c(...) * 53` are computed from the label distribution (Eq. 2 of the paper).

See **[Experimental_Setup.md](Experimental_Setup.md)** for the complete, reproducible experimental setup with all hyperparameters, data paths, preprocessing details, and a full list of conflicts between the paper and the code.

## Training

Run each model with its paper-reported hyperparameters (set `--optuna_trials 0`) or launch the Optuna search (default 50 trials):

```bash
# Ours (HybridPointTransformer) — Table 2 config hard-coded in Train_Hybrid.py
# (250 epochs, num_points = 65536; set num_points = 30000 in the config to reproduce the 30,000-point run)
python Train_Hybrid.py

# PointNet (Focal / AdamW / 2.4e-5 / 2.1e-5 / batch 1)
python Train_PointNet.py --optuna_trials 0 --epochs 150

# PointNet++ (Focal / AdamW / 5.38e-4 / 1e-6)
python Train_PointNetPP.py --optuna_trials 0 --epochs 150

# DGCNN (Focal / AdamW / 1.1e-5 / 4.2e-5)
python Train_Cust_DGCNN.py --optuna_trials 0 --epochs 150

# GNN / TeethGNN (Cross-Entropy / Adam / 9.6e-5 / 8e-6) — requires the graph cache first
python TeethGnnPreprocess_dataset.py
python Train_TeethGnn.py --optuna_trials 0 --epochs 150

# KPConv (Cross-Entropy / AdamW / 2.8e-5 / 7.49e-4)
python Train_kpconv.py --optuna_trials 0 --epochs 150
```

All scripts accept `--save_path`, `--experiment_name`, `--gpu_ids`, and `--epochs`. The Optuna search space follows the paper: learning rate `1e-6 .. 1e-2` (log), batch size `1`, optimizer `{Adam, AdamW}`, weight decay `1e-6 .. 1e-3` (log), focal gamma `1.0 .. 3.0`.

## Testing

The test scripts evaluate on the 10% held-out test split (`Dataset_900/test`) with `augment=False`, 65,536 points, and report overall accuracy and per-class mean IoU into `test_results/`:

```bash
python Test_Hybrid.py
python Test_PointNet.py
python Test_PointNetPP.py
python Test_Cust_DGCNN.py
python Test_kpconv.py
python Test_TeethGnn.py
```

`Test_TeethGnn.py` automatically builds the test graph cache on first run.

## Citation

If you use this code or the HybridPointTransformer in your research, please cite the paper (bibtex to be added).

## License

The code is released for research purposes. Please contact the authors for commercial use.