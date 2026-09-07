import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import optuna
from tqdm import tqdm
from torch.utils.data import random_split
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter
from torch_geometric.loader import DataLoader as GeometricDataLoader

# KPConv uses the paper-preprocessing dataloader (6-D input PyG Data records)
from dataloader.DataLoader_kpconv import ObjDataset
from networks.kpconv import KPConvNet

# General settings matching the paper
# Table 2 (Dental_Springer_BMC_Adjusted_v3): KPConv -> Cross-Entropy, AdamW,
#                                                          lr 2.8e-5, wd 7.49e-4, batch 1
config = {
    'data_path': 'K:/FromDevexy2024-5-py/PyRepos/segmentation/DeepLock/Dataset_600/train_data/data',
    'gt_path': 'K:/FromDevexy2024-5-py/PyRepos/segmentation/DeepLock/Dataset_600/train_data/ground-truth',
    'save_path': './checkpoints/KPConv_6D_600_final/',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'use_amp': torch.cuda.is_available(),
    'num_classes': 53,
    'binary': False,
    'val_split': 0.2,
    'test_split': 0.1,  # 70/20/10 = train/val/test
    'num_workers': 2,
    'num_points': 65536,  # paper: 65,536 points
    'epochs': 150,  # paper: 100-250
    'in_channels': 6,  # paper: 6-D input (xyz + rgb)

    'lr': 2.8e-5,  # Table 2
    'batch_size': 1,  # Table 2
    'weight_decay': 7.49e-4,  # Table 2
}

EARLY_STOPPING_PATIENCE = 10
MIN_EPOCHS = 150


def get_args():
    parser = argparse.ArgumentParser(description='Training KPConv (Optuna) for Dental Segmentation')
    parser.add_argument('--save_path', default='./experiments/', type=str, help='Path to save experiments')
    parser.add_argument('--experiment_name', default='KPConv_Optuna', type=str, help='Experiment name')
    parser.add_argument('--gpu_ids', default='0', type=str, help='GPU IDs')
    parser.add_argument('--optuna_trials', default=50, type=int, help='Number of Optuna trials')
    parser.add_argument('--epochs', default=150, type=int, help='Number of epochs for final training')
    return parser.parse_args()


def calculate_iou(pred, target, num_classes):
    """Calculate mean Intersection over Union (IoU) for each class."""
    ious = []
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)

    return np.nanmean(ious)


class EarlyStopping:
    def __init__(self, patience=EARLY_STOPPING_PATIENCE, min_epochs=MIN_EPOCHS):
        self.patience = patience
        self.min_epochs = min_epochs
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def __call__(self, val_value, epoch, maximize=False):
        if epoch < self.min_epochs:
            return False

        better = (val_value > self.best_value) if maximize else (val_value < self.best_value)
        if self.best_value is None:
            self.best_value = val_value
        elif not better:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_value = val_value
            self.counter = 0
        return False


def train(model, train_loader, loss_fn, optimizer, scheduler, epoch, device, scaler, num_classes):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    total_iou = 0
    num_batches = len(train_loader)

    with tqdm(train_loader, desc=f'Training Epoch {epoch}') as pbar:
        for batch_idx, data in enumerate(pbar):
            data = data.to(device)

            with autocast(enabled=config['use_amp']):
                output = model(data)  # [B*N, C]
                loss = loss_fn(output, data.y)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            target = data.y.view(-1)
            total += target.numel()
            correct += (predicted == target).sum().item()
            total_iou += calculate_iou(predicted, target, num_classes)

            pbar.set_postfix({
                'Loss': f'{total_loss / (batch_idx + 1):.4f}',
                'Acc': f'{100. * correct / total:.2f}%',
                'IoU': f'{total_iou / (batch_idx + 1):.4f}'
            })

    return total_loss / num_batches, 100. * correct / total, total_iou / num_batches


def validate(model, val_loader, loss_fn, device, num_classes):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    total_iou = 0
    num_batches = len(val_loader)

    with torch.no_grad():
        with tqdm(val_loader, desc='Validation') as pbar:
            for batch_idx, data in enumerate(pbar):
                data = data.to(device)

                output = model(data)
                target = data.y.view(-1)
                loss = loss_fn(output, target)

                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.numel()
                correct += (predicted == target).sum().item()
                total_iou += calculate_iou(predicted, target, num_classes)

                pbar.set_postfix({
                    'Loss': f'{total_loss / (batch_idx + 1):.4f}',
                    'Acc': f'{100. * correct / total:.2f}%',
                    'IoU': f'{total_iou / (batch_idx + 1):.4f}'
                })

    return total_loss / num_batches, 100. * correct / total, total_iou / num_batches


def build_loaders(num_points, batch_size):
    dataset = ObjDataset(config['data_path'], config['gt_path'],
                         num_points=num_points, num_classes=config['num_classes'],
                         augment=True)
    train_size = int((1 - config['val_split'] - config['test_split']) * len(dataset))
    val_size = int(config['val_split'] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size])

    train_loader = GeometricDataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=config['num_workers'], pin_memory=True, drop_last=True)
    val_loader = GeometricDataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=config['num_workers'], pin_memory=True, drop_last=True)
    return dataset, train_loader, val_loader


def objective(trial, args, device):
    # Search space (matching the paper: KPConv was optimised with Cross-Entropy, AdamW)
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1])
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.7)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    print(f"\n===== TRIAL {trial.number} STARTED =====")
    print(f"Hyperparameters being tested:")
    print(f"  lr={lr:.6f}, batch_size={batch_size}, dropout_rate={dropout_rate:.4f}, "
          f"optimizer={optimizer_name}, weight_decay={weight_decay:.6f}")
    print(f"Trials completed so far: {trial.number}")

    dataset, train_loader, val_loader = build_loaders(config['num_points'], batch_size)

    model = KPConvNet(in_channels=config['in_channels'], out_channels=64,
                      num_classes=config['num_classes'], debug=False,
                      dropout_rate=dropout_rate).to(device)
    scaler = GradScaler(enabled=config['use_amp'])

    if optimizer_name == "Adam":
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    loss_fn = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping()

    best_val_iou = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc, train_iou = train(model, train_loader, loss_fn, optimizer,
                                                 scheduler, epoch, device, scaler, config['num_classes'])
        val_loss, val_acc, val_iou = validate(model, val_loader, loss_fn, device, config['num_classes'])
        scheduler.step()

        trial.report(val_iou, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if early_stopping(val_iou, epoch, maximize=True):
            break

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(),
                       os.path.join(args.save_path, f'best_model_trial_600_{trial.number}.pth'))

    print(f"===== TRIAL {trial.number} FINISHED: best val mIoU = {best_val_iou:.4f} =====")
    return best_val_iou


def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    os.makedirs(args.save_path, exist_ok=True)

    device = config['device']
    print(f"Using device: {device}")

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.MedianPruner())

    best_params = None
    if args.optuna_trials > 0:
        def progress_callback(study, trial):
            done = len(study.trials)
            print(f"\n[PROGRESS] Trial {trial.number + 1}/{args.optuna_trials} completed "
                  f"({done}/{args.optuna_trials} so far) | best val mIoU so far: {study.best_value:.4f}")

        study.optimize(lambda trial: objective(trial, args, device),
                       n_trials=args.optuna_trials,
                       callbacks=[progress_callback])

        print("Number of finished trials:", len(study.trials))
        print("Best trial:")
        trial = study.best_trial
        print("  Value:", trial.value)
        print("  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        best_params = study.best_trial.params
    else:
        # Final training directly with the paper-reported optimized hyperparameters (Table 2)
        print("--optuna_trials=0 -> using paper-optimized hyperparameters (Table 2)")
        best_params = {
            'lr': config['lr'],
            'batch_size': config['batch_size'],
            'dropout_rate': 0.5,
            'optimizer': 'AdamW',
            'weight_decay': config['weight_decay'],
        }

    final_model = KPConvNet(in_channels=config['in_channels'], out_channels=64,
                            num_classes=config['num_classes'], debug=False,
                            dropout_rate=best_params['dropout_rate']).to(device)

    # Paper reports batch size 1 for every model (Table 2 / Section 6.1)
    final_batch_size = config['batch_size']
    dataset, train_loader, val_loader = build_loaders(config['num_points'], final_batch_size)

    if best_params['optimizer'] == "Adam":
        final_optimizer = Adam(final_model.parameters(), lr=best_params['lr'],
                               weight_decay=best_params['weight_decay'])
    else:
        final_optimizer = AdamW(final_model.parameters(), lr=best_params['lr'],
                                weight_decay=best_params['weight_decay'])

    final_scheduler = CosineAnnealingWarmRestarts(final_optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    # Paper: KPConv is trained with plain Cross-Entropy (Table 2)
    final_loss_fn = nn.CrossEntropyLoss()
    final_scaler = GradScaler(enabled=config['use_amp'])

    writer = SummaryWriter(os.path.join(args.save_path, args.experiment_name + '_final'))

    best_val_iou = 0.0
    early_stopping = EarlyStopping()

    for epoch in range(args.epochs):
        train_loss, train_acc, train_iou = train(final_model, train_loader, final_loss_fn,
                                                 final_optimizer, final_scheduler, epoch,
                                                 device, final_scaler, config['num_classes'])
        val_loss, val_acc, val_iou = validate(final_model, val_loader, final_loss_fn,
                                              device, config['num_classes'])
        final_scheduler.step()

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('IoU/train', train_iou, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('IoU/val', val_iou, epoch)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(final_model.state_dict(),
                       os.path.join(args.save_path, 'best_model_final_600.pth'))

        if early_stopping(val_iou, epoch, maximize=True):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    writer.close()


if __name__ == "__main__":
    main()