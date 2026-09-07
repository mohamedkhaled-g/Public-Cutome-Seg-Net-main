import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import your custom modules
from dataloader.DataLoader_HybridTransformer import ObjDataset
from networks.HybridTransformer import HybridPointTransformer


class PerSampleAccuracyTester:
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.model = None
        # Define consistent color mapping for dental classes
        self.color_map = self._create_consistent_color_map()

    def _create_consistent_color_map(self):
        """Create a consistent color mapping for dental classes."""
        # Create a color map that ensures same class gets same color
        # For 35 classes (0-34), we'll use tab20 with extensions
        colors = []

        # Use tab20 colors and repeat/modify for 35 classes
        tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
        tab20_bold = plt.cm.tab20b(np.linspace(0, 1, 20))

        for i in range(35):
            if i < 20:
                colors.append(tab20[i])
            elif i < 35:
                colors.append(tab20_bold[i % 20])
            else:
                # Fallback for any additional classes
                colors.append([0.5, 0.5, 0.5, 1.0])  # Gray

        return np.array(colors)

    def load_model(self, checkpoint_path):
        """Load model from checkpoint."""
        self.model = HybridPointTransformer(
            num_classes=self.config['num_classes'],
            d_model=self.config['d_model'],
            nhead=self.config['nhead']
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def compute_sample_metrics(self, pred, target):
        """Compute accuracy for a single sample."""
        pred_classes = pred.argmax(dim=-1).view(-1)
        target_flat = target.view(-1)
        accuracy = (pred_classes == target_flat).float().mean().item()
        return accuracy

    def visualize_single_sample(self, sample_idx, dataset, save_path=None):
        """Visualize ground truth vs predictions for a single sample with color verification."""
        sample = dataset[sample_idx]
        pos = sample['pos'].unsqueeze(0).to(self.device)
        x = sample['x'].unsqueeze(0).to(self.device)
        y = sample['y'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(pos, x)
            predictions = outputs.argmax(dim=-1).squeeze(0).cpu().numpy()

        ground_truth = y.squeeze(0).cpu().numpy()
        positions = pos.squeeze(0).cpu().numpy()

        # Get original filename
        obj_file = dataset.obj_files[sample_idx]
        sample_name = os.path.basename(obj_file)

        # Create figure with subplots (enhanced layout)
        fig = plt.figure(figsize=(28, 7))

        # Ground Truth
        ax1 = fig.add_subplot(151, projection='3d')
        scatter1 = ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                               c=ground_truth, cmap='tab20', s=1, alpha=0.8)
        ax1.set_title(f'Ground Truth\n{sample_name}', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # Predictions
        ax2 = fig.add_subplot(152, projection='3d')
        scatter2 = ax2.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                               c=predictions, cmap='tab20', s=1, alpha=0.8)
        ax2.set_title(f'Predictions\n{sample_name}', fontsize=12, fontweight='bold')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        # Differences (Errors)
        differences = predictions != ground_truth
        error_colors = np.where(differences, 'red', 'lightblue')
        ax3 = fig.add_subplot(153, projection='3d')
        scatter3 = ax3.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                               c=error_colors, s=1, alpha=0.8)
        ax3.set_title(f'Prediction Errors\n(Red = Wrong, Blue = Correct)',
                      fontsize=12, fontweight='bold')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')

        # Color consistency verification
        ax4 = fig.add_subplot(154)
        ax4.axis('off')

        # Show unique classes and their colors
        unique_gt = np.unique(ground_truth)
        unique_pred = np.unique(predictions)
        all_unique = np.unique(np.concatenate([unique_gt, unique_pred]))

        # Show color mapping
        for i, cls in enumerate(all_unique[:20]):  # Show first 20 classes
            color_idx = cls % 20  # Use tab20 colormap
            color = plt.cm.tab20(color_idx)
            ax4.scatter(i, 0, c=[color], s=100, label=f'Class {cls}')

        ax4.set_title('Color Mapping\n(Same class = Same color)', fontsize=12)
        ax4.set_xlim(-0.5, min(19.5, len(all_unique) - 0.5))
        ax4.set_ylim(-0.5, 0.5)
        ax4.set_yticks([])

        # Accuracy and statistics
        ax5 = fig.add_subplot(155)
        ax5.axis('off')

        # Calculate statistics
        accuracy = (predictions == ground_truth).mean()
        unique_classes_gt = len(unique_gt)
        unique_classes_pred = len(unique_pred)
        error_rate = np.mean(differences)

        stats_text = f"Sample: {sample_name}\n"
        stats_text += f"Accuracy: {accuracy:.4f}\n"
        stats_text += f"Error Rate: {error_rate:.4f}\n"
        stats_text += f"GT Classes: {unique_classes_gt}\n"
        stats_text += f"Pred Classes: {unique_classes_pred}\n"
        stats_text += f"Total Points: {len(ground_truth)}"

        ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"Visualization saved to: {save_path}")

        # Close the figure to free memory (don't display it)
        plt.close(fig)

        return sample_name

    def test_all_samples_and_save(self, checkpoint_path, accuracy_file_path='./sample_accuracies.txt',
                                  max_samples=None, visualize_samples=True,
                                  visualization_interval=10):
        """
        Test EACH sample individually, save accuracy + filename to file, and visualize samples.

        Parameters:
        - checkpoint_path: Path to model checkpoint
        - accuracy_file_path: File to save individual sample accuracies with filenames
        - max_samples: Maximum number of samples to test (None for all)
        - visualize_samples: If True, creates visualizations for samples
        - visualization_interval: Visualize every Nth sample
        """
        # Load model
        self.load_model(checkpoint_path)

        # Load dataset
        dataset = ObjDataset(
            obj_dir=self.config['data_path'],
            json_dir=self.config['gt_path'],
            num_points=self.config['num_points'],
            augment=False,
            binary=self.config['binary'],
            num_classes=self.config['num_classes']
        )

        # Determine number of samples to test
        if max_samples is None:
            total_samples = len(dataset)
        else:
            total_samples = min(max_samples, len(dataset))

        print(f"Testing {total_samples} samples...")

        # Create visualization directory
        vis_dir = os.path.join(os.path.dirname(accuracy_file_path), 'sample_visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        # Open file to write accuracies with filenames
        with open(accuracy_file_path, 'w') as acc_file:
            acc_file.write("# Sample Index, Filename, Accuracy\n")

        all_accuracies = []
        all_filenames = []

        for sample_idx in range(total_samples):
            # Get single sample
            sample = dataset[sample_idx]
            pos = sample['pos'].unsqueeze(0).to(self.device)
            x = sample['x'].unsqueeze(0).to(self.device)
            y = sample['y'].unsqueeze(0).to(self.device)

            # Get prediction
            with torch.no_grad():
                outputs = self.model(pos, x)
                accuracy = self.compute_sample_metrics(outputs, y)

            # Store accuracy and filename
            all_accuracies.append(accuracy)

            # Extract filename from dataset
            obj_file = dataset.obj_files[sample_idx]
            sample_name = os.path.basename(obj_file)

            # Write to file: index, filename, accuracy
            with open(accuracy_file_path, 'a') as acc_file:
                acc_file.write(f"{sample_idx}, {sample_name}, {accuracy:.6f}\n")

            # Print progress
            if (sample_idx + 1) % 10 == 0:
                print(f"Processed {sample_idx + 1}/{total_samples}, {sample_name}: {accuracy:.6f}")

            # Visualize sample if it's the right interval
            if visualize_samples and (sample_idx % visualization_interval == 0):
                vis_path = os.path.join(vis_dir, f'{sample_name.replace(".obj", "_visualization.png")}')
                print(f"Visualizing sample {sample_idx} ({sample_name})...")
                self.visualize_single_sample(sample_idx, dataset, save_path=vis_path)

        # Calculate and append summary statistics
        mean_accuracy = np.mean(all_accuracies)
        std_accuracy = np.std(all_accuracies)
        min_accuracy = np.min(all_accuracies)
        max_accuracy = np.max(all_accuracies)
        median_accuracy = np.median(all_accuracies)

        with open(accuracy_file_path, 'a') as acc_file:
            acc_file.write(f"\n# SUMMARY STATISTICS\n")
            acc_file.write(f"# Mean Accuracy: {mean_accuracy:.6f}\n")
            acc_file.write(f"# Std Accuracy: {std_accuracy:.6f}\n")
            acc_file.write(f"# Min Accuracy: {min_accuracy:.6f}\n")
            acc_file.write(f"# Max Accuracy: {max_accuracy:.6f}\n")
            acc_file.write(f"# Median Accuracy: {median_accuracy:.6f}\n")
            acc_file.write(f"# Total Samples: {len(all_accuracies)}\n")

        print(f"\nTesting completed!")
        print(f"Mean Accuracy: {mean_accuracy:.6f}")
        print(f"Std Accuracy: {std_accuracy:.6f}")
        print(f"Min Accuracy: {min_accuracy:.6f}")
        print(f"Max Accuracy: {max_accuracy:.6f}")
        print(f"Median Accuracy: {median_accuracy:.6f}")
        print(f"All individual sample accuracies saved to: {accuracy_file_path}")

        return all_accuracies, mean_accuracy

    def test_specific_samples(self, checkpoint_path, sample_indices,
                              accuracy_file_path='./specific_sample_accuracies.txt',
                              visualize_samples=True):
        """
        Test specific samples by their indices, including filename in output.

        Parameters:
        - sample_indices: List of sample indices to test
        """
        # Load model
        self.load_model(checkpoint_path)

        # Load dataset
        dataset = ObjDataset(
            obj_dir=self.config['data_path'],
            json_dir=self.config['gt_path'],
            num_points=self.config['num_points'],
            augment=False,
            binary=self.config['binary'],
            num_classes=self.config['num_classes']
        )

        # Create visualization directory
        vis_dir = os.path.join(os.path.dirname(accuracy_file_path), 'specific_sample_visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        # Open file to write accuracies with filenames
        with open(accuracy_file_path, 'w') as acc_file:
            acc_file.write("# Sample Index, Filename, Accuracy\n")

        all_accuracies = []

        for sample_idx in sample_indices:
            if sample_idx >= len(dataset):
                print(f"Warning: Sample index {sample_idx} is out of range. Skipping.")
                continue

            # Get single sample
            sample = dataset[sample_idx]
            pos = sample['pos'].unsqueeze(0).to(self.device)
            x = sample['x'].unsqueeze(0).to(self.device)
            y = sample['y'].unsqueeze(0).to(self.device)

            # Get prediction
            with torch.no_grad():
                outputs = self.model(pos, x)
                accuracy = self.compute_sample_metrics(outputs, y)

            # Store accuracy
            all_accuracies.append(accuracy)

            # Extract filename from dataset
            obj_file = dataset.obj_files[sample_idx]
            sample_name = os.path.basename(obj_file)

            # Write to file: index, filename, accuracy
            with open(accuracy_file_path, 'a') as acc_file:
                acc_file.write(f"{sample_idx}, {sample_name}, {accuracy:.6f}\n")

            print(f"Sample {sample_idx} ({sample_name}): {accuracy:.6f}")

            # Visualize sample
            if visualize_samples:
                vis_path = os.path.join(vis_dir, f'{sample_name.replace(".obj", "_visualization.png")}')
                print(f"Visualizing sample {sample_idx} ({sample_name})...")
                self.visualize_single_sample(sample_idx, dataset, save_path=vis_path)

        # Calculate and append summary statistics
        if all_accuracies:
            mean_accuracy = np.mean(all_accuracies)
            std_accuracy = np.std(all_accuracies)
            min_accuracy = np.min(all_accuracies)
            max_accuracy = np.max(all_accuracies)
            median_accuracy = np.median(all_accuracies)

            with open(accuracy_file_path, 'a') as acc_file:
                acc_file.write(f"\n# SUMMARY STATISTICS FOR SPECIFIC SAMPLES\n")
                acc_file.write(f"# Mean Accuracy: {mean_accuracy:.6f}\n")
                acc_file.write(f"# Std Accuracy: {std_accuracy:.6f}\n")
                acc_file.write(f"# Min Accuracy: {min_accuracy:.6f}\n")
                acc_file.write(f"# Max Accuracy: {max_accuracy:.6f}\n")
                acc_file.write(f"# Median Accuracy: {median_accuracy:.6f}\n")
                acc_file.write(f"# Total Samples: {len(all_accuracies)}\n")

        print(f"\nSpecific samples testing completed!")
        print(f"All individual sample accuracies saved to: {accuracy_file_path}")

        return all_accuracies


def main():
    config = {
        'data_path': 'K:/FromDevexy2024-5-py/PyRepos/segmentation/DeepLock/Dataset_600/test_data/data',
        'gt_path': 'K:/FromDevexy2024-5-py/PyRepos/segmentation/DeepLock/Dataset_600/test_data/ground-truth',
        'checkpoint_path': './checkpoints/HybridTransformer_Final_6D_600_try4/final_hybrid_6D_model.pth',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'num_classes': 35,
        'binary': False,
        'num_workers': 4,
        'num_points': 65536,
        'batch_size': 1,
        'd_model': 256,
        'nhead': 4,
    }

    tester = PerSampleAccuracyTester(config)

    # Test samples with filenames and visualizations
    print("Testing samples with filenames and saving individual accuracies...")
    all_accuracies, mean_acc = tester.test_all_samples_and_save(
        checkpoint_path=config['checkpoint_path'],
        accuracy_file_path='./test_results/sample_accuracies_with_filenames.txt',
        max_samples=1000,
        visualize_samples=True,
        visualization_interval=1  # Visualize every sample
    )

    print(f"\nTesting completed! Mean accuracy: {mean_acc:.6f}")
    print("All visualizations have been saved to the 'sample_visualizations' folder.")
    print("No figures were displayed - you can check the saved PNG files directly.")


if __name__ == "__main__":
    # Add this line to ensure matplotlib doesn't try to display figures
    plt.ioff()
    main()