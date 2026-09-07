# Train_TeethGnn.py
import os
import argparse
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import SubsetRandomSampler
from torch_geometric.loader import DataLoader

import optuna
from networks.TeethGnn_Model import TeethGNN
from dataloader.Dataloader_TeethGnn_Cached import TeethGNNDatasetCached

warnings.filterwarnings('ignore', category=UserWarning)

# General settings matching the paper
# Table 2 (Dental_Springer_BMC_Adjusted_v3): GNN (TeethGNN) -> Cross-Entropy, Adam,
#                                                        lr 9.6e-5, wd 8e-6, batch 1
config = {
    'processed_data_path': './TeethGnn_processed_data_53',
    'save_path': './experiments/TeethGnn_Optuna_final/',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'use_amp': torch.cuda.is_available(),
    'num_classes': 53,  # FDI 53-class general setting
    'binary': False,
    'val_split': 0.2,
    'test_split': 0.1,  # 70/20/10 = train/val/test
    'num_workers': 2,
    'epochs': 150,  # paper: 100-250
    'batch_size': 1,  # Table 2: batch size 1
    'model_name': 'TeethGnn_600',

    # Paper-optimized hyperparameters (Table 2)
    'lr': 9.6e-5,
    'weight_decay': 8e-6,
}


def get_args():
    parser = argparse.ArgumentParser(description='Training TeethGNN (Optuna) for Dental Segmentation')
    parser.add_argument('--save_path', default='./experiments/', type=str, help='Path to save experiments')
    parser.add_argument('--experiment_name', default='TeethGnn_Optuna', type=str, help='Experiment name')
    parser.add_argument('--gpu_ids', default='0', type=str, help='GPU IDs')
    parser.add_argument('--optuna_trials', default=50, type=int, help='Number of Optuna trials')
    parser.add_argument('--epochs', default=150, type=int, help='Number of epochs for final training')
    return parser.parse_args()


def compute_metrics(pred, target, num_classes=53):
    """Compute per-class accuracy and mean IoU. Works for graph node predictions."""
    pred = pred.argmax(dim=-1)
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    # Ignore out-of-range / missing labels
    valid = target < num_classes
    pred = pred[valid]
    target = target[valid]

    if target.numel() == 0:
        return 0.0, 0.0

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


def train_model(trial, config, train_dataset, val_dataset):
    # Hyperparameter search
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True) if trial else config['lr']
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True) if trial else config['weight_decay']

    if trial:
        print(f"\n===== TRIAL {trial.number} STARTED =====")
        print("Hyperparameters being tested:")
        print(f"  lr={lr:.6f}, weight_decay={weight_decay:.6f}")
        print(f"Trials completed so far: {trial.number}")

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )

    model = TeethGNN(num_classes=config['num_classes']).to(config['device'])

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    scaler = GradScaler(enabled=config['use_amp'])

    # Paper: GNN baseline uses plain Cross-Entropy (Table 2)
    criterion = nn.CrossEntropyLoss()

    best_val_mIoU = 0.0
    early_stop_counter = 0
    patience = 10

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_miou = 0.0
        epoch_start = time.time()

        for data in train_loader:
            data = data.to(config['device'])

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=config['use_amp']):
                sem_logits, _, _ = model(data)
                loss = criterion(sem_logits, data.y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            acc, mIoU = compute_metrics(sem_logits, data.y, config['num_classes'])
            train_loss += loss.item()
            train_acc += acc
            train_miou += mIoU

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_miou = 0.0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(config['device'])

                with autocast(enabled=config['use_amp']):
                    sem_logits, _, _ = model(data)
                    loss = criterion(sem_logits, data.y)

                acc, mIoU = compute_metrics(sem_logits, data.y, config['num_classes'])
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

    run_config = dict(config)
    run_config['save_path'] = args.save_path
    run_config['epochs'] = args.epochs
    run_config['optuna_trials'] = args.optuna_trials

    print("Configuration:")
    for k, v in run_config.items():
        print(f"  {k}: {v}")

    # 70/20/10 split. Preprocessing is training-only (Algorithm 1): the training
    # split gets the +/-10 deg / [0.9, 1.1] augmentation, the validation split does not.
    print(f"Loading cached dataset from {run_config['processed_data_path']} ...")
    full_dataset = TeethGNNDatasetCached(run_config['processed_data_path'],
                                         num_classes=run_config['num_classes'],
                                         augment=False)
    indices = np.arange(len(full_dataset))
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    n_train = int((1 - run_config['val_split'] - run_config['test_split']) * len(full_dataset))
    n_val = int(run_config['val_split'] * len(full_dataset))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]

    train_dataset = TeethGNNDatasetCached(run_config['processed_data_path'],
                                          num_classes=run_config['num_classes'],
                                          augment=True)
    train_dataset.file_list = [full_dataset.file_list[i] for i in train_idx]
    val_dataset = TeethGNNDatasetCached(run_config['processed_data_path'],
                                        num_classes=run_config['num_classes'],
                                        augment=False)
    val_dataset.file_list = [full_dataset.file_list[i] for i in val_idx]

    if run_config['optuna_trials'] > 0:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
        )

        def progress_callback(study, trial):
            done = len(study.trials)
            print(f"\n[PROGRESS] Trial {trial.number + 1}/{run_config['optuna_trials']} completed "
                  f"({done}/{run_config['optuna_trials']} so far) | best val mIoU so far: {study.best_value:.4f}")

        study.optimize(
            lambda trial: train_model(trial, run_config, train_dataset, val_dataset),
            n_trials=run_config['optuna_trials'],
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
        # Final training directly with the paper-reported optimized hyperparameters (Table 2)
        print("--optuna_trials=0 -> training with paper-optimized hyperparameters (Table 2)")
        train_model(None, run_config, train_dataset, val_dataset)


if __name__ == '__main__':
    main()