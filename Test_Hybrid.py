import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader.DataLoader_HybridTransformer import ObjDataset
from networks.HybridTransformer import HybridPointTransformer

DATASET_ROOT = './Dataset_900'

config = {
    'data_path': f'{DATASET_ROOT}/test/data',
    'gt_path': f'{DATASET_ROOT}/test/ground-truth',
    'checkpoint_path': './checkpoints/HybridTransformer_Final_6D_600_try5_53/checkpoint_best.pth',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'num_classes': 53,
    'num_points': 65536,
    'batch_size': 1,
    'num_workers': 2,
    'd_model': 256,
    'nhead': 4,
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
    print(f"Testing HybridPointTransformer (Ours) on the 10% test split (Dataset_900/test)")
    device = config['device']
    model = HybridPointTransformer(
        num_classes=config['num_classes'],
        d_model=config['d_model'],
        nhead=config['nhead'],
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
        binary=False,
        num_classes=config['num_classes'],
    )
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False,
                        num_workers=config['num_workers'])

    total_acc, total_miou, count = 0.0, 0.0, 0
    rows = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            pos, x, y = batch['pos'], batch['x'], batch['y']
            logits = model(pos, x)   
            acc, miou = compute_metrics(logits, y, config['num_classes'])
            total_acc += acc
            total_miou += miou
            count += 1
            rows.append([os.path.basename(dataset.obj_files[i]), acc, miou])

    final_acc = total_acc / count
    final_miou = total_miou / count
    print(f"\nTest Accuracy: {final_acc:.4f}")
    print(f"Test mIoU:     {final_miou:.4f}")

    os.makedirs('./test_results', exist_ok=True)
    with open('./test_results/Hybrid_test_results.txt', 'w') as f:
        f.write(f"Accuracy: {final_acc:.6f}\n")
        f.write(f"mIoU: {final_miou:.6f}\n")
    with open('./test_results/Hybrid_test_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample', 'accuracy', 'mIoU'])
        writer.writerows(rows)
    print("Results saved to ./test_results/Hybrid_test_results.{txt,csv}")

if __name__ == '__main__':
    main()
