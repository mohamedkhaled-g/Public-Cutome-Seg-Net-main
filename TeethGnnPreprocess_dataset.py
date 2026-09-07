# preprocess_dataset.py
import os
import torch
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from dataloader.Dataloader_TeethGnn import TeethGNNDataset

# Configuration
config = {
    'data_path': 'K:/FromDevexy2024-5-py/PyRepos/segmentation/DeepLock/Dataset_600/train_data/data',
    'gt_path': 'K:/FromDevexy2024-5-py/PyRepos/segmentation/DeepLock/Dataset_600/train_data/ground-truth',
    'num_facets': 10000,
    'num_classes': 53,
    'output_dir': './TeethGnn_processed_data_53'
}

_dataset = None


def _init_worker():
    global _dataset
    _dataset = TeethGNNDataset(
        obj_dir=config['data_path'],
        json_dir=config['gt_path'],
        num_facets=config['num_facets'],
        num_classes=config['num_classes']
    )


def _process_one(idx):
    global _dataset
    filepath = os.path.join(config['output_dir'], f"{idx:06d}.pt")
    if os.path.exists(filepath):
        return idx, True  # already done (resume)
    data = _dataset[idx]
    torch.save(data, filepath)
    return idx, False


def main():
    parser = argparse.ArgumentParser(description='TeethGNN graph preprocessing (parallel, 53-class)')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--max', type=int, default=0, help='0 = process all samples; >0 caps the count')
    parser.add_argument('--output_dir', type=str, default=config['output_dir'])
    args = parser.parse_args()
    config['output_dir'] = args.output_dir
    config['num_classes'] = 53

    os.makedirs(config['output_dir'], exist_ok=True)

    probe = TeethGNNDataset(
        obj_dir=config['data_path'],
        json_dir=config['gt_path'],
        num_facets=config['num_facets'],
        num_classes=config['num_classes']
    )
    total = len(probe)
    print(f"Total samples available: {total}")

    indices = list(range(min(total, args.max))) if args.max > 0 else list(range(total))

    print(f"Starting preprocessing of {len(indices)} samples with {args.workers} workers...")
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        futures = {pool.submit(_process_one, i): i for i in indices}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                idx, was_cached = fut.result()
                done += 1
                status = "cached" if was_cached else "done"
                if done % 20 == 0 or done == len(indices):
                    print(f"Preprocessed {done}/{len(indices)} samples.")
            except Exception as e:
                print(f"Error processing sample {i}: {e}")

    print("Preprocessing complete!")


if __name__ == '__main__':
    main()