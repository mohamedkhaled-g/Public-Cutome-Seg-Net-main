import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from torch.cuda.amp import GradScaler, autocast
import optuna
from collections import defaultdict
import time
import warnings

from dataloader.DataLoader_Cust_DGCNN import ObjDataset
from networks.Cust_DGCNN_Dilated import DGCNN_Dilated

warnings.filterwarnings('ignore', category=UserWarning)

# General settings matching the paper
# Table 2 (Dental_Springer_BMC_Adjusted_v3): DGCNN -> Focal, AdamW, lr 1.1e-5, wd 4.2e-5, batch 1
config = {
    'data_path': 'K:/FromDevexy2024-5-py/PyRepos/segmentation/DeepLock/Dataset_600/train_data/data',
    'gt_path': 'K:/FromDevexy2024-5-py/PyRepos/segmentation/DeepLock/Dataset_600/train_data/ground-truth',
    'save_path': './experiments/DGCNN_Optuna_final/',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'use_amp': torch.cuda.is_available(),
    'num_classes': 53,
    'binary': False,
    'val_split': 0.2,
    'test_split': 0.1,  # 70/20/10 = train/val/test
    'num_workers': 2,
    'num_points': 65536,
    'epochs': 150,  # paper: 100-250
    'batch_size': 1,  # Table 2: batch size 1
    'model_name': 'DGCNN_Dilated_6D_600',

    # Paper-optimized hyperparameters (Table 2)
    'lr': 1.1e-5,
    'weight_decay': 4.2e-5,
    'focal_gamma': 2.8214,
}


class FocalLoss(nn.Module):
    """FocalLoss matching the HybridPointTransformer (Ours) loss."""
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()


def get_args():
    parser = argparse.ArgumentParser(description='Training DGCNN (Optuna) for Dental Segmentation')
    parser.add_argument('--save_path', default='./experiments/', type=str, help='Path to save experiments')
    parser.add_argument('--experiment_name', default='DGCNN_Optuna', type=str, help='Experiment name')
    parser.add_argument('--gpu_ids', default='0', type=str, help='GPU IDs')
    parser.add_argument('--optuna_trials', default=50, type=int, help='Number of Optuna trials')
    parser.add_argument('--epochs', default=150, type=int, help='Number of epochs for final training')
    return parser.parse_args()


def compute_metrics(pred, target, num_classes=53):
    """Compute per-class accuracy and mean IoU."""
    pred = pred.argmax(dim=len(pred.shape) - 1)
    correct = (pred == target).float().sum()
    accuracy = correct / target.numel()

    iou = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).float().sum()
        union = (pred_cls | target_cls).float().sum()
        iou.append(intersection / (union + 1e-6))

    mIoU = np.mean(iou)
    return accuracy.item(), mIoU


def train_model(trial, config):
    # Hyperparameter search
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True) if trial else config['lr']
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True) if trial else config['weight_decay']
    k_local = trial.suggest_int("k_local", 15, 30) if trial else 20
    k_dilated = [
        trial.suggest_int("k_dilated_1", 50, 200, step=50) if trial else 100,
        trial.suggest_int("k_dilated_2", 100, 300, step=50) if trial else 200,
        trial.suggest_int("k_dilated_3", 200, 400, step=50) if trial else 300
    ]
    focal_gamma = trial.suggest_float("focal_gamma", 1.0, 3.0) if trial else config['focal_gamma']

    if trial:
        print(f"\n===== TRIAL {trial.number} STARTED =====")
        print("Hyperparameters being tested:")
        print(f"  lr={lr:.6f}, weight_decay={weight_decay:.6f}, k_local={k_local}, "
              f"k_dilated={k_dilated}, focal_gamma={focal_gamma:.4f}")
        print(f"Trials completed so far: {trial.number}")

    dataset = ObjDataset(
        obj_dir=config['data_path'],
        json_dir=config['gt_path'],
        num_points=config['num_points'],
        augment=True,
        num_classes=config['num_classes']
    )

    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    n_train = int((1 - config['val_split'] - config['test_split']) * len(dataset))
    n_val = int(config['val_split'] * len(dataset))

    train_loader = DataLoader(
        dataset, batch_size=config['batch_size'],
        sampler=SubsetRandomSampler(indices[:n_train]),
        num_workers=config['num_workers'], pin_memory=True
    )
    val_loader = DataLoader(
        dataset, batch_size=config['batch_size'],
        sampler=SubsetRandomSampler(indices[n_train:n_train + n_val]),
        num_workers=config['num_workers'], pin_memory=True
    )

    model = DGCNN_Dilated(
        num_classes=config['num_classes'],
        k_local=k_local,
        k_dilated=k_dilated
    ).to(config['device'])

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    scaler = GradScaler(enabled=config['use_amp'])

    class_weights = dataset.get_label_weights().to(config['device'])
    criterion = FocalLoss(gamma=focal_gamma, alpha=class_weights)

    best_val_mIoU = 0.0
    early_stop_counter = 0
    patience = 10

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_miou = 0.0
        epoch_start = time.time()

        for batch in train_loader:
            vertices = batch['vertices'].to(config['device'], non_blocking=True)
            labels = batch['labels'].to(config['device'], non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=config['use_amp']):
                outputs = model(vertices)  # [B, N, C]
                loss = criterion(outputs.reshape(-1, config['num_classes']), labels.reshape(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            acc, mIoU = compute_metrics(outputs, labels, config['num_classes'])
            train_loss += loss.item()
            train_acc += acc
            train_miou += mIoU

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_miou = 0.0

        with torch.no_grad():
            for batch in val_loader:
                vertices = batch['vertices'].to(config['device'], non_blocking=True)
                labels = batch['labels'].to(config['device'], non_blocking=True)

                with autocast(enabled=config['use_amp']):
                    outputs = model(vertices)
                    loss = criterion(outputs.reshape(-1, config['num_classes']), labels.reshape(-1))

                acc, mIoU = compute_metrics(outputs, labels, config['num_classes'])
                val_loss += loss.item()
                val_acc += acc
                val_miou += mIoU

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1}/{config['epochs']} ({epoch_time:.1f}s):")
        print(f"  Train Loss: {train_loss / len(train_loader):.4f} | "
              f"Acc: {train_acc / len(train_loader) * 100:.2f}% | "
              f"mIoU: {train_miou / len(train_loader):.4f}")
        print(f"  Val Loss: {val_loss / len(val_loader):.4f} | "
              f"Acc: {val_acc / len(val_loader) * 100:.2f}% | "
              f"mIoU: {val_miou / len(val_loader):.4f}")

        current_mIoU = val_miou / len(val_loader)
        if trial:
            trial.report(current_mIoU, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if current_mIoU > best_val_mIoU:
            best_val_mIoU = current_mIoU
            early_stop_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(config['save_path'], f'{config["model_name"]}_best.pth'))
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if trial:
        print(f"===== TRIAL {trial.number} FINISHED: best val mIoU = {best_val_mIoU:.4f} =====")
    return best_val_mIoU


def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    os.makedirs(args.save_path, exist_ok=True)

    # Prepare a config copy with run settings for this run
    run_config = dict(config)
    run_config['save_path'] = args.save_path
    run_config['epochs'] = args.epochs

    print("Configuration:")
    for k, v in run_config.items():
        print(f"  {k}: {v}")

    if args.optuna_trials > 0:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
        )

        def progress_callback(study, trial):
            done = len(study.trials)
            print(f"\n[PROGRESS] Trial {trial.number + 1}/{args.optuna_trials} completed "
                  f"({done}/{args.optuna_trials} so far) | best val mIoU so far: {study.best_value:.4f}")

        study.optimize(
            lambda trial: train_model(trial, run_config),
            n_trials=args.optuna_trials,
            callbacks=[progress_callback]
        )

        print("Number of finished trials:", len(study.trials))
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value (mIoU): {trial.value:.4f}")
        print("  Params:")
        for k, v in trial.params.items():
            print(f"    {k}: {v}")
    else:
        # Final training directly with the paper-optimized hyperparameters (Table 2)
        print("--optuna_trials=0 -> training with paper-optimized hyperparameters (Table 2)")
        train_model(None, run_config)


if __name__ == '__main__':
    main()
