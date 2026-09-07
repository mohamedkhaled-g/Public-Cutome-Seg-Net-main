import os
import csv
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from dataloader.DataLoader_kpconv import ObjDataset
from networks.kpconv import KPConvNet

DATASET_ROOT = 'K:/FromDevexy2024-5-py/PyRepos/segmentation/DeepLock/Dataset_900'

config = {
    'data_path': f'{DATASET_ROOT}/test/data',
    'gt_path': f'{DATASET_ROOT}/test/ground-truth',
    'checkpoint_path': './checkpoints/KPConv_6D_600_final/best_model_final_600.pth',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'num_classes': 53,
    'in_channels': 6,
    'out_channels': 64,
    'num_points': 65536,
    'batch_size': 1,
    'num_workers': 2,
}


def calculate_iou(pred, target, num_classes):
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


def main():
    print("Testing KPConv on the 10% test split (Dataset_900/test)")
    device = config['device']
    model = KPConvNet(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        num_classes=config['num_classes'],
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

    all_iou = []
    total_correct, total_elements = 0, 0
    count = 0
    rows = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(device)
            logits = model(data)            # [B*N, C]
            target = data.y                 # [B*N]
            pred = logits.argmax(dim=-1)
            correct = (pred == target).float().sum().item()
            total_correct += correct
            total_elements += target.numel()
            iou = calculate_iou(logits, target, config['num_classes'])
            all_iou.append(iou)
            count += 1
            rows.append([f'sample_{i:04d}', correct / target.numel(), float(np.mean(iou))])

    final_acc = total_correct / total_elements if total_elements > 0 else 0.0
    final_miou = float(np.mean(all_iou)) if all_iou else 0.0
    print(f"\nTest Accuracy: {final_acc:.4f}")
    print(f"Test mIoU:     {final_miou:.4f}")

    os.makedirs('./test_results', exist_ok=True)
    with open('./test_results/KPConv_test_results.txt', 'w') as f:
        f.write(f"Accuracy: {final_acc:.6f}\n")
        f.write(f"mIoU: {final_miou:.6f}\n")
    with open('./test_results/KPConv_test_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample', 'accuracy', 'mIoU'])
        writer.writerows(rows)
    print("Results saved to ./test_results/KPConv_test_results.{txt,csv}")


if __name__ == '__main__':
    main()