import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from dataloader.DataLoader_HybridTransformer import ObjDataset
from networks.HybridTransformer import HybridPointTransformer

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

class DentalSegmentationTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.best_miou = 0.0
        os.makedirs(config['save_path'], exist_ok=True)

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.early_stop_counter = 0
        self.patience = config.get('early_stopping_patience', 20)

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

    def run_epoch(self, model, loader, criterion, optimizer, is_train, accumulation_steps=1):
        model.train(is_train)
        total_loss, total_acc, total_iou = 0.0, 0.0, 0.0
        desc = "Training" if is_train else "Validation"

        if is_train:
            optimizer.zero_grad()

        for i, batch in enumerate(tqdm(loader, desc=desc, leave=False)):
            pos = batch['pos'].to(self.device)
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)

            with torch.set_grad_enabled(is_train):
                outputs = model(pos, x)
                loss = criterion(outputs.reshape(-1, self.config['num_classes']), y.reshape(-1))

            if is_train:
                loss = loss / accumulation_steps
                loss.backward()

                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
                    optimizer.step()
                    optimizer.zero_grad()

            acc, iou = self.compute_metrics(outputs, y)
            total_loss += loss.item() * accumulation_steps
            total_acc += acc
            total_iou += iou

        return {'loss': total_loss / len(loader), 'accuracy': total_acc / len(loader), 'mIoU': total_iou / len(loader)}

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"⚠️ No checkpoint found at {checkpoint_path}. Starting from scratch.")
            return 0

        print(f"🔁 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model weights loaded successfully")

        if 'best_miou' in checkpoint:
            self.best_miou = checkpoint['best_miou']
            print(f"✅ Loaded best mIoU: {self.best_miou:.4f}")

        if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✅ Optimizer state loaded")

        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"✅ Resuming from epoch {start_epoch}")
            return start_epoch

        print("⚠️ No epoch info in checkpoint. Starting from epoch 1.")
        return 1

    def _initialize_components(self, dataset, load_checkpoint=True):
        self.model = HybridPointTransformer(
            num_classes=self.config['num_classes'],
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            debug=self.config.get('debug_model', False)
        ).to(self.device)

        class_weights = dataset.get_label_weights().to(self.device)
        self.criterion = FocalLoss(gamma=self.config['focal_gamma'], alpha=class_weights)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config['lr'],
                                     weight_decay=self.config['weight_decay'])

        
        if load_checkpoint:
            checkpoint_path = os.path.join(self.config['save_path'], 'checkpoint_best.pth')
            start_epoch = self.load_checkpoint(checkpoint_path)
            return start_epoch
        return 1

    def train(self):
        dataset = ObjDataset(
            obj_dir=self.config['data_path'],
            json_dir=self.config['gt_path'],
            num_points=self.config['num_points'],
            augment=True,
            binary=self.config['binary'],
            num_classes=self.config['num_classes']
        )

        indices = list(range(len(dataset)))
        np.random.shuffle(indices)
        n_train = int((1 - self.config['val_split'] - self.config['test_split']) * len(dataset))
        n_val = int(self.config['val_split'] * len(dataset))
        train_indices, val_indices = indices[:n_train], indices[n_train:n_train + n_val]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(dataset, batch_size=self.config['batch_size'], sampler=train_sampler,
                                  num_workers=self.config['num_workers'], pin_memory=False, drop_last=True)
        val_loader = DataLoader(dataset, batch_size=self.config['batch_size'], sampler=val_sampler,
                                num_workers=self.config['num_workers'], pin_memory=False, drop_last=True)

        
        log_file = os.path.join(self.config['save_path'], 'training_log.csv')
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Train_Loss', 'Train_Acc', 'Train_mIoU', 'Val_Loss', 'Val_Acc', 'Val_mIoU'])
        else:
            print(f"📊 Continuing existing log file: {log_file}")

        
        start_epoch = self._initialize_components(dataset, load_checkpoint=True)

        print("--- CONTINUING Training ---")
        print(f"Using device: {self.device}")
        print(f"Total steps per epoch: {len(train_loader)} | Training epochs: {start_epoch} to {self.config['epochs']}")
        print(f"Total additional epochs: {self.config['epochs'] - start_epoch + 1}")

        
        print("\n🔍 Quick validation to verify model state...")
        with torch.no_grad():
            sample_batch = next(iter(val_loader))
            pos = sample_batch['pos'].to(self.device)
            x = sample_batch['x'].to(self.device)
            sample_output = self.model(pos, x)
            sample_pred = sample_output.argmax(dim=-1)
            sample_acc = (sample_pred == sample_batch['y'].to(self.device)).float().mean().item()
            print(f"Sample validation accuracy: {sample_acc:.4f}")

        for epoch in range(start_epoch, self.config['epochs']):
            print(f"\n--- Epoch {epoch + 1}/{self.config['epochs']} ---")

            train_metrics = self.run_epoch(self.model, train_loader, self.criterion, self.optimizer, is_train=True,
                                           accumulation_steps=self.config['accumulation_steps'])
            print(
                f"Train | Loss: {train_metrics['loss']:.6f} | Acc: {train_metrics['accuracy']:.4f} | mIoU: {train_metrics['mIoU']:.4f}")

            val_metrics = self.run_epoch(self.model, val_loader, self.criterion, None, is_train=False,
                                         accumulation_steps=1)
            print(
                f"Val   | Loss: {val_metrics['loss']:.6f} | Acc: {val_metrics['accuracy']:.4f} | mIoU: {val_metrics['mIoU']:.4f}")

            
            is_new_best = val_metrics['mIoU'] > self.best_miou

            if is_new_best:
                self.best_miou = val_metrics['mIoU']
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            if self.early_stop_counter >= self.patience:
                print(f"🛑 Early stopping triggered after {epoch + 1} epochs (no improvement for {self.patience} epochs)")
                break

            
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    round(train_metrics['loss'], 6),
                    round(train_metrics['accuracy'], 6),
                    round(train_metrics['mIoU'], 6),
                    round(val_metrics['loss'], 6),
                    round(val_metrics['accuracy'], 6),
                    round(val_metrics['mIoU'], 6)
                ])

            
            if is_new_best:
                print(f"🚀 New best model found with mIoU: {self.best_miou:.4f}! Saving to {self.config['save_path']}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_miou': self.best_miou,
                }, os.path.join(self.config['save_path'], 'checkpoint_best.pth'))

            
            if (epoch + 1) % 5 == 0:
                backup_path = os.path.join(self.config['save_path'], f'backup_epoch_{epoch + 1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_miou': self.best_miou,
                }, backup_path)
                print(f"💾 Backup saved: {backup_path}")

        print(f"\n--- TRAINING COMPLETED ---")
        print(f"Final best validation mIoU: {self.best_miou:.4f}")
        print(f"Training completed from epoch {start_epoch} to {self.config['epochs']}")

def main():
    
    
    config = {
        'data_path': './Dataset_900/train/data',
        'gt_path': './Dataset_900/train/ground-truth',
        'save_path': './checkpoints/HybridTransformer_Final_6D_600_try5_53/',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'use_amp': torch.cuda.is_available(),
        'num_classes': 53,
        'binary': False,
        'val_split': 0.2,
        'test_split': 0.1,  
        'epochs': 250,  
        'accumulation_steps': 1,  

        'num_workers': 2,
        'num_points': 65536,
        'lr': 1e-4,
        'batch_size': 1,
        'weight_decay': 1e-4,
        'focal_gamma': 2.8214,
        'd_model': 256,
        'nhead': 4,
        'early_stopping_patience': 20,
        'debug_model': False,
    }

    print("--- CONTINUING Training from Last Checkpoint (53 classes) ---")
    print(f"Using device: {config['device']}")
    print(f"Number of classes: {config['num_classes']}")

    trainer = DentalSegmentationTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
