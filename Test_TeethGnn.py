import os
import csv
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from dataloader.Dataloader_TeethGnn import TeethGNNDataset
from dataloader.Dataloader_TeethGnn_Cached import TeethGNNDatasetCached
from networks.TeethGnn_Model import TeethGNN

DATASET_ROOT = 'K:/FromDevexy2024-5-py/PyRepos/segmentation/DeepLock/Dataset_900'

config = {
    'data_path': f'{DATASET_ROOT}/test/data',
    'gt_path': f'{DATASET_ROOT}/test/ground-truth',
    'processed_dir': './TeethGnn_processed_data_900_test',
    'checkpoint_path': './experiments/TeethGnn_600_best.pth',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'num_classes': 53,
    'num_facets': 10000,
    'batch_size': 1,
    'num_workers': 2,
}


def calculate_iou(pred, target, num_classes):
    pred = pred.argmax(dim=-1) if len(pred.shape) > 1 else pred
    iou = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = (pred_cls & target_cls).float().sum().item()
        union = (pred_cls | target_cls).float().sum().item()
        iou.append(intersection / (union + 1e-6))
    return iou


def build_test_cache():
    proc_dir = config['processed_dir']
    os.makedirs(proc_dir, exist_ok=True)
    if any(f.endswith('.pt') for f in os.listdir(proc_dir)):
        print(f"Test cache already exists in {proc_dir} — skipping build.")
        return

    print(f"Building test-set cache → {proc_dir}")
    ds = TeethGNNDataset(
        obj_dir=config['data_path'],
        json_dir=config['gt_path'],
        num_facets=config['num_facets'],
        num_classes=config['num_classes'],
    )
    for i in range(len(ds)):
        data = ds[i]
        torch.save(data, os.path.join(proc_dir, f'{i:06d}.pt'))
        if (i + 1) % 10 == 0 or (i + 1) == len(ds):
            print(f"  cached {i + 1}/{len(ds)}")
    print("Test cache build complete.")


def main():
    print("Testing TeethGNN on the 10% test split (Dataset_900/test)")
    build_test_cache()
    device = config['device']

    model = TeethGNN(num_classes=config['num_classes']).to(device)
    checkpoint = torch.load(config['checkpoint_path'], map_location=device)
    state = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    print(f"Checkpoint loaded: {config['checkpoint_path']}")

    dataset = TeethGNNDatasetCached(processed_dir=config['processed_dir'],
                                    num_classes=config['num_classes'],
                                    augment=False)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False,
                        num_workers=config['num_workers'])

    all_iou = []
    total_correct, total_elements = 0, 0
    rows = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(device)
            sem_logits, offsets, pos = model(data)  # (N, C), (N, 3), (N, 3)
            target = data.y                          # (N,)
            pred = sem_logits.argmax(dim=-1)
            correct = (pred == target).float().sum().item()
            total_correct += correct
            total_elements += target.numel()
            iou = calculate_iou(sem_logits, target, config['num_classes'])
            all_iou.append(iou)
            rows.append([f'sample_{i:04d}', correct / target.numel(), float(np.mean(iou))])

    final_acc = total_correct / total_elements if total_elements > 0 else 0.0
    final_miou = float(np.mean(all_iou)) if all_iou else 0.0
    print(f"\nTest Accuracy: {final_acc:.4f}")
    print(f"Test mIoU:     {final_miou:.4f}")

    os.makedirs('./test_results', exist_ok=True)
    with open('./test_results/TeethGnn_test_results.txt', 'w') as f:
        f.write(f"Accuracy: {final_acc:.6f}\n")
        f.write(f"mIoU: {final_miou:.6f}\n")
    with open('./test_results/TeethGnn_test_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample', 'accuracy', 'mIoU'])
        writer.writerows(rows)
    print("Results saved to ./test_results/TeethGnn_test_results.{txt,csv}")


if __name__ == '__main__':
    main()