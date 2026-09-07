import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dataloader.DataLoader_DentalMAN import ObjDataset_Arch
from networks.DentalMAN import DentalMAN


class AccuracyOnlyTester:
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.model = None

    def load_model(self, checkpoint_path):
        self.model = DentalMAN(
            num_classes=self.config['num_classes'],
            d_model=self.config['d_model'],
            n_heads=self.config['nhead'],
            num_points=self.config['num_points'],
            dropout=self.config.get('dropout', 0.1)
        ).to(self.device)

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        self.model.load_state_dict(state_dict)
        self.model.eval()

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

    def visualize_sample(self, sample_idx, dataset, save_path=None):
        sample = dataset[sample_idx]
        pos = sample['pos'].unsqueeze(0).to(self.device)
        x = sample['x'].unsqueeze(0).to(self.device)
        y = sample['y'].unsqueeze(0).to(self.device)
        arc = sample.get('arc_params', None)
        if arc is not None:
            arc = arc.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(pos, x, arc)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            predictions = outputs.argmax(dim=-1).squeeze(0).cpu().numpy()

        ground_truth = y.squeeze(0).cpu().numpy()
        positions = pos.squeeze(0).cpu().numpy()

        fig = plt.figure(figsize=(20, 6))

        ax1 = fig.add_subplot(131, projection='3d')
        scatter1 = ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                               c=ground_truth, cmap='tab20', s=1, alpha=0.8)
        ax1.set_title('Ground Truth', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        ax2 = fig.add_subplot(132, projection='3d')
        scatter2 = ax2.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                               c=predictions, cmap='tab20', s=1, alpha=0.8)
        ax2.set_title('Predictions', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        differences = predictions != ground_truth
        error_colors = np.where(differences, 'red', 'lightblue')
        ax3 = fig.add_subplot(133, projection='3d')
        scatter3 = ax3.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                               c=error_colors, s=1, alpha=0.8)
        ax3.set_title('Prediction Errors\n(Red = Wrong, Blue = Correct)',
                      fontsize=14, fontweight='bold')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"Visualization saved to: {save_path}")

        plt.show()

    def test_and_save_accuracy(self, checkpoint_path, test_results_path='./test_accuracy.txt',
                               test_on_entire_dataset=True, max_samples=None,
                               visualize_samples=True, sample_to_visualize=None):
        self.load_model(checkpoint_path)

        dataset = ObjDataset_Arch(
            obj_dir=self.config['data_path'],
            json_dir=self.config['gt_path'],
            num_points=self.config['num_points'],
            augment=False,
            binary=self.config['binary'],
            num_classes=self.config['num_classes']
        )

        if test_on_entire_dataset:
            test_dataset = dataset
            print(f"Testing on ENTIRE dataset: {len(dataset)} samples")
        else:
            if max_samples is None:
                max_samples = 100
            max_samples = min(max_samples, len(dataset))
            test_dataset = Subset(dataset, list(range(max_samples)))
            print(f"Testing on SUBSET: {max_samples} samples")

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers']
        )

        total_acc = 0.0
        total_iou = 0.0
        count = 0

        with torch.no_grad():
            for batch in test_loader:
                pos = batch['pos'].to(self.device)
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)
                arc = batch.get('arc_params', None)
                if arc is not None:
                    arc = arc.to(self.device)

                outputs = self.model(pos, x, arc)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                acc, iou = self.compute_metrics(outputs, y)

                total_acc += acc
                total_iou += iou
                count += 1

        final_accuracy = total_acc / count
        final_miou = total_iou / count

        with open(test_results_path, 'w') as f:
            f.write(f"{final_accuracy:.6f}\n")

        print(f"Final Accuracy: {final_accuracy:.6f}")
        print(f"Final mIoU: {final_miou:.6f}")
        print(f"Accuracy saved to: {test_results_path}")

        vis_dir = os.path.dirname(test_results_path) if os.path.dirname(test_results_path) else '.'
        vis_dir = os.path.join(vis_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        if visualize_samples:
            if sample_to_visualize is None:
                sample_to_visualize = 0

            if sample_to_visualize >= len(dataset):
                print(f"Warning: Sample index {sample_to_visualize} is out of range. "
                      f"Dataset has {len(dataset)} samples.")
                sample_to_visualize = len(dataset) - 1

            vis_path = os.path.join(vis_dir, f'sample_{sample_to_visualize}_comparison.png')
            print(f"\nVisualizing sample {sample_to_visualize}...")
            self.visualize_sample(sample_to_visualize, dataset, save_path=vis_path)

        return final_accuracy


def main():
    config = {
        'data_path': 'K:/FromDevexy2024-5-py/PyRepos/segmentation/DeepLock/Dataset_600/train_data/data',
        'gt_path': 'K:/FromDevexy2024-5-py/PyRepos/segmentation/DeepLock/Dataset_600/train_data/ground-truth',
        'checkpoint_path': './checkpoints/DentalMAN/dentalman_best_model.pth',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'num_classes': 53,
        'binary': False,
        'num_workers': 4,
        'num_points': 8192,
        'batch_size': 1,
        'd_model': 256,
        'nhead': 4,
        'dropout': 0.1,
    }

    tester = AccuracyOnlyTester(config)

    accuracy = tester.test_and_save_accuracy(
        checkpoint_path=config['checkpoint_path'],
        test_results_path='./test_results/dentalman_accuracy_results.txt',
        test_on_entire_dataset=False,
        max_samples=100,
        visualize_samples=True,
        sample_to_visualize=48
    )


if __name__ == "__main__":
    main()
