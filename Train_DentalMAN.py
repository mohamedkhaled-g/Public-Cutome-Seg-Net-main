import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import torch.nn.functional as F

from dataloader.DataLoader_DentalMAN import ObjDataset_Arch
from networks.DentalMAN import DentalMAN


class FocalLoss(nn.Module):
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


class DentalMAN_Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.best_miou = 0.0
        os.makedirs(config['save_path'], exist_ok=True)

    def compute_metrics(self, pred, target):
        pred_classes = pred.argmax(dim=-1).view(-1)
        target_flat = target.view(-1)
        accuracy = (pred_classes == target_flat).float().mean().item()
        iou_per_class = []
        for cls in range(self.config['num_classes']):
            pred_inds = (pred_classes == cls)
            target_inds = (target_flat == cls)
            if target_inds.sum() > 0:
                intersection = (pred_inds[target_inds]).long().sum().item()
                union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
                iou_per_class.append(intersection / (union + 1e-8))
        return accuracy, np.mean(iou_per_class) if iou_per_class else 0.0

    def run_epoch(self, model, loader, criterion, optimizer, scaler, is_train):
        model.train(is_train)
        total_loss, total_acc, total_iou = 0.0, 0.0, 0.0
        desc = "Training" if is_train else "Validation"
        for batch in tqdm(loader, desc=desc, leave=False):
            pos = batch['pos'].to(self.device)
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            arc_params = batch.get('arc_params', None)
            if arc_params is not None:
                arc_params = arc_params.to(self.device)

            with torch.set_grad_enabled(is_train):
                with autocast(enabled=self.config['use_amp']):
                    outputs = model(pos, x, arc_params)
                    if isinstance(outputs, tuple):
                        main_out, aux2, aux3 = outputs
                        loss_main = criterion(main_out.reshape(-1, self.config['num_classes']), y.reshape(-1))
                        loss_aux2 = criterion(aux2.reshape(-1, self.config['num_classes']), y.reshape(-1))
                        loss_aux3 = criterion(aux3.reshape(-1, self.config['num_classes']), y.reshape(-1))
                        loss = loss_main + 0.4 * loss_aux2 + 0.3 * loss_aux3
                    else:
                        loss = criterion(outputs.reshape(-1, self.config['num_classes']), y.reshape(-1))

            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if isinstance(outputs, tuple):
                out_for_metrics = outputs[0]
            else:
                out_for_metrics = outputs

            acc, iou = self.compute_metrics(out_for_metrics, y)
            total_loss += loss.item()
            total_acc += acc
            total_iou += iou

        return {'loss': total_loss / len(loader), 'accuracy': total_acc / len(loader), 'mIoU': total_iou / len(loader)}

    def train(self):
        dataset = ObjDataset_Arch(
            obj_dir=self.config['data_path'],
            json_dir=self.config['gt_path'],
            num_points=self.config['num_points'],
            augment=True,
            binary=self.config['binary'],
            num_classes=self.config['num_classes']
        )

        indices = list(range(len(dataset)))
        np.random.shuffle(indices)
        split = int(np.floor(self.config['val_split'] * len(dataset)))
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(dataset, batch_size=self.config['batch_size'], sampler=train_sampler,
                                  num_workers=self.config['num_workers'], pin_memory=True)
        val_loader = DataLoader(dataset, batch_size=self.config['batch_size'], sampler=val_sampler,
                                num_workers=self.config['num_workers'], pin_memory=True)

        model = DentalMAN(
            num_classes=self.config['num_classes'],
            d_model=self.config['d_model'],
            n_heads=self.config['nhead'],
            num_points=self.config['num_points'],
            dropout=self.config.get('dropout', 0.1)
        ).to(self.device)

        class_weights = dataset.get_label_weights().to(self.device)
        criterion = FocalLoss(gamma=self.config['focal_gamma'], alpha=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=self.config['lr'],
                                weight_decay=self.config['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
        scaler = GradScaler(enabled=self.config['use_amp'])

        for epoch in range(self.config['epochs']):
            print(f"\n--- Epoch {epoch + 1}/{self.config['epochs']} ---")
            train_metrics = self.run_epoch(model, train_loader, criterion, optimizer, scaler, is_train=True)
            print(f"Train | Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} "
                  f"| mIoU: {train_metrics['mIoU']:.4f}")
            val_metrics = self.run_epoch(model, val_loader, criterion, None, scaler, is_train=False)
            print(f"Val   | Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} "
                  f"| mIoU: {val_metrics['mIoU']:.4f}")
            scheduler.step()

            if val_metrics['mIoU'] > self.best_miou:
                self.best_miou = val_metrics['mIoU']
                print(f"New best model found with mIoU: {self.best_miou:.4f}! "
                      f"Saving to {self.config['save_path']}")
                torch.save(model.state_dict(),
                           os.path.join(self.config['save_path'], 'dentalman_best_model.pth'))

        print(f"\n--- FINAL TRAINING FINISHED ---")
        print(f"Best validation mIoU achieved: {self.best_miou:.4f}")


def main():
    config = {
        'data_path': 'K:/FromDevexy2024-5-py/PyRepos/segmentation/DeepLock/Dataset_100/train_data/data',
        'gt_path': 'K:/FromDevexy2024-5-py/PyRepos/segmentation/DeepLock/Dataset_100/train_data/ground-truth',
        'save_path': './checkpoints/DentalMAN/',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'use_amp': torch.cuda.is_available(),
        'num_classes': 53,
        'binary': False,
        'val_split': 0.2,
        'num_workers': 4,

        'num_points': 32768,
        'epochs': 50,

        'lr': 1e-3,
        'batch_size': 4,
        'weight_decay': 1e-4,
        'focal_gamma': 2.0,
        'd_model': 256,
        'nhead': 4,
        'dropout': 0.1,
    }

    print("--- Starting DentalMAN Training (53 FDI classes, 6D input) ---")
    print(f"Using device: {config['device']}")

    trainer = DentalMAN_Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
