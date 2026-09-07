import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import torch.nn.functional as F

# Import your custom modules
from dataloader.DataLoader_SPT import ObjDataset
from networks.SuperpointTransformer import SuperpointTransformer


class FocalLoss(nn.Module):
    """
    Focal Loss, as defined in https://arxiv.org/abs/1708.02002
    """

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
            # This part is crucial for class weighting
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


class DentalSegmentationTrainer:
    """
    Manages the entire training and validation process.
    """

    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.best_miou = 0.0
        os.makedirs(config['save_path'], exist_ok=True)

    def compute_metrics(self, pred, target):
        """Calculates accuracy and mean Intersection-over-Union (mIoU)."""
        pred_classes = pred.argmax(dim=-1).view(-1)
        target_flat = target.view(-1)

        accuracy = (pred_classes == target_flat).float().mean().item()

        iou_per_class = []
        # Iterate over all possible classes
        for cls in range(self.config['num_classes']):
            pred_inds = (pred_classes == cls)
            target_inds = (target_flat == cls)

            # Only calculate IoU for classes that are actually present in the ground truth
            if target_inds.sum() > 0:
                intersection = (pred_inds[target_inds]).long().sum().item()
                union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
                # Add a small epsilon to avoid division by zero
                iou_per_class.append(intersection / (union + 1e-8))

        mean_iou = np.mean(iou_per_class) if iou_per_class else 0.0
        return accuracy, mean_iou

    def run_epoch(self, model, loader, criterion, optimizer, scaler, is_train):
        """Runs a single epoch of either training or validation."""
        model.train(is_train)
        total_loss, total_acc, total_iou = 0.0, 0.0, 0.0

        desc = "Training" if is_train else "Validation"
        for batch in tqdm(loader, desc=desc, leave=False):
            # Move data to the configured device (GPU/CPU)
            vertices = batch['vertices'].to(self.device)
            labels = batch['labels'].to(self.device)
            sp_labels = batch['superpoint_labels'].to(self.device)
            sp_centers = batch['superpoint_centers'].to(self.device)

            # Enable/disable gradients for efficiency
            with torch.set_grad_enabled(is_train):
                # Use Automatic Mixed Precision for faster training on compatible GPUs
                with autocast(enabled=self.config['use_amp']):
                    outputs = model(vertices, sp_labels, sp_centers)
                    loss = criterion(outputs.view(-1, self.config['num_classes']), labels.view(-1))

            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            acc, iou = self.compute_metrics(outputs, labels)
            total_loss += loss.item()
            total_acc += acc
            total_iou += iou

        avg_loss = total_loss / len(loader)
        avg_acc = total_acc / len(loader)
        avg_iou = total_iou / len(loader)
        return {'loss': avg_loss, 'accuracy': avg_acc, 'mIoU': avg_iou}

    def train(self):
        """The main training loop."""
        # --- 1. Dataset and DataLoaders ---
        dataset = ObjDataset(
            obj_dir=self.config['data_path'],
            json_dir=self.config['gt_path'],
            num_points=self.config['num_points'],
            binary=self.config['binary'],
            num_classes=self.config['num_classes']
        )

        indices = list(range(len(dataset)))
        np.random.shuffle(indices)
        split = int(np.floor(self.config['val_split'] * len(dataset)))
        train_indices, val_indices = indices[split:], indices[:split]

        train_loader = DataLoader(dataset, batch_size=self.config['batch_size'],
                                  sampler=SubsetRandomSampler(train_indices), num_workers=self.config['num_workers'],
                                  pin_memory=True)
        val_loader = DataLoader(dataset, batch_size=self.config['batch_size'], sampler=SubsetRandomSampler(val_indices),
                                num_workers=self.config['num_workers'], pin_memory=True)

        # --- 2. Model ---
        model = SuperpointTransformer(
            num_classes=self.config['num_classes'],
            nhead=self.config['nhead'],
            d_model=self.config['d_model']
        ).to(self.device)

        # --- 3. Loss, Optimizer, and Scheduler ---
        class_weights = dataset.get_label_weights().to(self.device)
        criterion = FocalLoss(gamma=self.config['focal_gamma'], alpha=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['epochs'], eta_min=1e-6)
        scaler = GradScaler(enabled=self.config['use_amp'])

        # --- 4. Training Loop ---
        for epoch in range(self.config['epochs']):
            print(f"\n--- Epoch {epoch + 1}/{self.config['epochs']} ---")

            train_metrics = self.run_epoch(model, train_loader, criterion, optimizer, scaler, is_train=True)
            print(
                f"Train | Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | mIoU: {train_metrics['mIoU']:.4f}")

            val_metrics = self.run_epoch(model, val_loader, criterion, None, scaler, is_train=False)
            print(
                f"Val   | Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | mIoU: {val_metrics['mIoU']:.4f}")

            scheduler.step()

            # Save the model only if it has the best mIoU so far
            if val_metrics['mIoU'] > self.best_miou:
                self.best_miou = val_metrics['mIoU']
                print(f"🚀 New best model found with mIoU: {self.best_miou:.4f}! Saving to {self.config['save_path']}")
                torch.save(model.state_dict(), os.path.join(self.config['save_path'], 'final_best_model.pth'))

        print(f"\n--- Training Finished ---")
        print(f"Best validation mIoU achieved: {self.best_miou:.4f}")


def main():
    # This configuration dictionary holds all the settings for the final run.
    # It uses the best parameters discovered by the Optuna hyperparameter search.
    config = {
        # --- File Paths and Device Config ---
        'data_path': 'K:/FromDevexy2024-5-py/PyRepos/segmentation/DeepLock/Dataset_100/train_data/data',
        'gt_path': 'K:/FromDevexy2024-5-py/PyRepos/segmentation/DeepLock/Dataset_100/train_data/ground-truth',
        'save_path': './checkpoints/SuperPointTransformerNet/V3',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'use_amp': torch.cuda.is_available(),

        # --- Core Task & Data Parameters ---
        'num_classes': 35,
        'binary': False,
        'num_points': 8192,
        'val_split': 0.2,
        'num_workers': 4,
        'd_model': 256,

        # --- Winning Hyperparameters from Optuna's Best Trial (Trial 8) ---
        'epochs': 200,  # ENHANCEMENT: Train for much longer to maximize performance
        'lr': 0.0019317,
        'batch_size': 2,
        'weight_decay': 2.5454e-05,
        'focal_gamma': 1.9043,
        'nhead': 4,
    }

    print("--- Starting FINAL Training Run ---")
    print("Using the best hyperparameters found by Optuna.")
    print(f"Parameters: {config}")

    trainer = DentalSegmentationTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
