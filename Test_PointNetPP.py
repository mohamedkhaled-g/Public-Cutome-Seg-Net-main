import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader.DataLoader_PP import ObjDataset
from networks.PointNetPlusPlus import PointNetPlusPlus

DATASET_ROOT = 'K:/FromDevexy2024-5-py/PyRepos/segmentation/DeepLock/Dataset_900'

config = {
    'data_path': f'{DATASET_ROOT}/test/data',
    'gt_path': f'{DATASET_ROOT}/test/ground-truth',
    'checkpoint_path': './experiments/PointNetPP_Optuna_final/best_model_final.pth',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'num_classes': 53,
    'in_channels': 3,
    'dropout_rate': 0.5,
    'num_points': 65536,
    'batch_size': 1,
    'num_workers': 2,
}


def compute_metrics(logits, target, num_classes=53):
    pred = logits.argmax(dim=-1).reshape(-1)
    target_flat = target.reshape(-1)
    accuracy = (pred == target_flat).float().mean().item()
    iou = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target_flat == cls
        intersection = (pred_cls & target_cls).float().sum().item()
        union = (pred_cls | target_cls).float().sum().item()
        iou.append(intersection / (union + 1e-6))
    return accuracy, float(np.mean(iou))


def main():
    print("Testing PointNet++ on the 10% test split (Dataset_900/test)")
    device = config['device']
    model = PointNetPlusPlus(
        num_classes=config['num_classes'],
        in_channels=config['in_channels'],
        dropout_rate=config['dropout_rate'],
    ).to(device)

    checkpoint = torch.load(config['checkpoint_path'], map_location=device)
    state = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    print(f"Checkpoint loaded: {config['checkpoint_path']}")

    dataset = ObjDataset(
        obj_dir=config['data_path'],
        json_dir=config['gt_path'],
        num_points=config['num_points'],
        augment=False,
        num_classes=config['num_classes'],
    )
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False,
                        num_workers=config['num_workers'])

    total_acc, total_miou, count = 0.0, 0.0, 0
    rows = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            vertices = batch['vertices'].to(device)
            labels = batch['labels'].to(device)
            x = vertices.transpose(1, 2)  # [B, 6, N]
            logits = model(x)             # [B, C, N]
            logits = logits.permute(0, 2, 1)
            acc, miou = compute_metrics(logits, labels, config['num_classes'])
            total_acc += acc
            total_miou += miou
            count += 1
            rows.append([os.path.basename(dataset.obj_files[i]), acc, miou])

    final_acc = total_acc / count
    final_miou = total_miou / count
    print(f"\nTest Accuracy: {final_acc:.4f}")
    print(f"Test mIoU:     {final_miou:.4f}")

    os.makedirs('./test_results', exist_ok=True)
    with open('./test_results/PointNetPP_test_results.txt', 'w') as f:
        f.write(f"Accuracy: {final_acc:.6f}\n")
        f.write(f"mIoU: {final_miou:.6f}\n")
    with open('./test_results/PointNetPP_test_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample', 'accuracy', 'mIoU'])
        writer.writerows(rows)
    print("Results saved to ./test_results/PointNetPP_test_results.{txt,csv}")


if __name__ == '__main__':
    main()