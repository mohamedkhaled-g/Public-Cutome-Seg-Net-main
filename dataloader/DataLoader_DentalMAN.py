import os
import numpy as np
import json
import torch
from torch.utils.data import Dataset
import random
from tqdm import tqdm
from torch_points3d.core.data_transform import ElasticDistortion
from torch_geometric.data import Data


def compute_arch_curve_params(points_3d, num_segments=40):
    """
    Fit an arch curve to the point cloud and compute arc-length parameters.

    Uses PCA projection + piecewise linear approximation along the arch
    (avoids polynomial oscillation issues of high-degree polyfit).

    Returns:
        arc_params: (N, 1) arc-length parameter in [0, 1]
    """
    points = points_3d.copy()
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    cov = centered.T @ centered / (centered.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_axes = eigvecs[:, ::-1]
    proj_2d = centered @ main_axes[:, :2]

    angles = np.arctan2(proj_2d[:, 1], proj_2d[:, 0])
    radii = np.sqrt(np.sum(proj_2d ** 2, axis=1))

    sort_idx = np.argsort(angles)
    angles_sorted = angles[sort_idx]
    radii_sorted = radii[sort_idx]

    if len(angles_sorted) < 10:
        return np.linspace(0, 1, len(points)).reshape(-1, 1).astype(np.float32)

    try:
        segment_boundaries = np.linspace(angles_sorted[0], angles_sorted[-1], num_segments + 1)
        segment_idx = np.digitize(angles_sorted, segment_boundaries) - 1
        segment_idx = np.clip(segment_idx, 0, num_segments - 1)
        segment_centers = np.array([
            angles_sorted[segment_idx == s].mean() if (segment_idx == s).any()
            else (segment_boundaries[s] + segment_boundaries[s + 1]) / 2
            for s in range(num_segments)
        ])
        segment_radii = np.array([
            radii_sorted[segment_idx == s].mean() if (segment_idx == s).any()
            else np.interp(segment_centers[s], angles_sorted, radii_sorted)
            for s in range(num_segments)
        ])

        theta_fine = segment_centers
        r_fine = np.maximum(segment_radii, 1e-8)

        dx = np.diff(r_fine * np.cos(theta_fine))
        dy = np.diff(r_fine * np.sin(theta_fine))
        segment_lengths = np.sqrt(dx ** 2 + dy ** 2)
        cumulative = np.zeros(len(theta_fine))
        cumulative[1:] = np.cumsum(segment_lengths)

        total_arc_length = cumulative[-1]
        if total_arc_length < 1e-8:
            arc_params_norm = np.linspace(0, 1, len(points))
            return arc_params_norm.reshape(-1, 1).astype(np.float32)

        arc_lengths_fine = cumulative / total_arc_length
        arc_params = np.interp(angles_sorted, theta_fine, arc_lengths_fine)
        arc_params_all = np.zeros(len(points))
        arc_params_all[sort_idx] = arc_params
    except np.linalg.LinAlgError:
        arc_params_all = np.linspace(0, 1, len(points))

    arc_params_all = np.clip(arc_params_all, 0, 1).reshape(-1, 1)
    return arc_params_all.astype(np.float32)


class ObjDataset_Arch(Dataset):
    def __init__(self, obj_dir, json_dir, num_points=8192, augment=True,
                 max_vertices=100000, binary=False, num_classes=53):
        self.obj_dir = obj_dir
        self.json_dir = json_dir
        self.num_points = num_points
        self.augment = augment
        self.max_vertices = max_vertices
        self.binary = binary
        self.num_classes = num_classes

        if self.augment:
            self.elastic_distortion = ElasticDistortion(
                apply_distorsion=True,
                granularity=[0.2, 0.8],
                magnitude=[0.4, 1.6]
            )

        self.obj_files = []
        self.json_files = []
        for root, _, files in os.walk(obj_dir):
            for file in files:
                if file.endswith('.obj'):
                    obj_path = os.path.join(root, file)
                    relative_path = os.path.relpath(obj_path, obj_dir)
                    json_path = os.path.join(json_dir, os.path.splitext(relative_path)[0] + '.json')
                    if os.path.exists(json_path):
                        self.obj_files.append(obj_path)
                        self.json_files.append(json_path)

        if not self.obj_files:
            raise ValueError("No valid .obj/.json pairs found in the provided directories")
        print(f"Found {len(self.obj_files)} valid .obj/.json pairs")
        self.label_weights = self._compute_label_weights()

    def _compute_label_weights(self):
        label_counts = np.zeros(self.num_classes)
        for json_file in tqdm(self.json_files, desc="Computing label weights"):
            try:
                labels = self._read_ground_truth(json_file)
                labels = self._remap_labels(labels)
                valid_mask = (labels >= 0) & (labels < self.num_classes)
                unique, counts = np.unique(labels[valid_mask], return_counts=True)
                label_counts[unique] += counts
            except Exception as e:
                print(f"Warning: Error processing {json_file} for weight calculation: {str(e)}")

        label_counts = np.where(label_counts == 0, 1, label_counts)
        weights = 1.0 / np.log(1.02 + label_counts)
        weights = weights / np.sum(weights) * self.num_classes
        print(f"Computed label weights for {self.num_classes} classes.")
        return torch.tensor(weights, dtype=torch.float32)

    def _read_obj_file(self, obj_path):
        vertices_and_colors = []
        with open(obj_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if parts and parts[0] == 'v':
                    try:
                        vertices_and_colors.append(list(map(float, parts[1:7])))
                    except (ValueError, IndexError):
                        continue
        features = np.array(vertices_and_colors, dtype=np.float32)
        if len(features) == 0:
            raise ValueError(f"No vertices found in {obj_path}")
        return features

    def _read_ground_truth(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return np.array(data['labels'], dtype=np.int64)

    def _remap_labels(self, labels):
        remapped = np.full_like(labels, -1)
        remapped = np.where(labels == 0, 0, remapped)

        for fdi in range(11, 19):
            remapped = np.where(labels == fdi, fdi - 10, remapped)
        for fdi in range(21, 29):
            remapped = np.where(labels == fdi, fdi - 20 + 8, remapped)
        for fdi in range(31, 39):
            remapped = np.where(labels == fdi, fdi - 30 + 16, remapped)
        for fdi in range(41, 49):
            remapped = np.where(labels == fdi, fdi - 40 + 24, remapped)

        for fdi in range(51, 56):
            remapped = np.where(labels == fdi, fdi - 50 + 32, remapped)
        for fdi in range(61, 66):
            remapped = np.where(labels == fdi, fdi - 60 + 37, remapped)
        for fdi in range(71, 76):
            remapped = np.where(labels == fdi, fdi - 70 + 42, remapped)
        for fdi in range(81, 86):
            remapped = np.where(labels == fdi, fdi - 80 + 47, remapped)

        unmapped_mask = remapped == -1
        if unmapped_mask.any():
            print(f"Warning: Found {unmapped_mask.sum()} labels that could not be remapped. "
                  f"Unique values: {np.unique(labels[unmapped_mask])}")

        return remapped

    def _compute_arch_params(self, features):
        pos = features[:, :3]
        return compute_arch_curve_params(pos)

    def _augment_features(self, features):
        pos = features[:, :3].copy()
        if random.random() > 0.5:
            angle = np.random.uniform(-np.pi / 18, np.pi / 18)
            rot_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                   [np.sin(angle), np.cos(angle), 0],
                                   [0, 0, 1]], dtype=np.float32)
            pos = pos @ rot_matrix.T
        if random.random() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            pos *= scale

        colors = features[:, 3:].copy()
        if random.random() > 0.5:
            colors += np.random.uniform(-0.1, 0.1, 3)
        colors = np.clip(colors, 0, 1)

        return np.hstack([pos, colors])

    def _normalize_pos(self, pos):
        centroid = np.mean(pos, axis=0)
        pos = pos - centroid
        max_dist = np.max(np.sqrt(np.sum(pos ** 2, axis=1)))
        if max_dist > 1e-8:
            pos = pos / max_dist
        return pos

    def __len__(self):
        return len(self.obj_files)

    def __getitem__(self, idx):
        obj_path = self.obj_files[idx]
        json_path = self.json_files[idx]

        try:
            features = self._read_obj_file(obj_path)
            labels = self._read_ground_truth(json_path)

            min_len = min(len(features), len(labels))
            features, labels = features[:min_len], labels[:min_len]

            if self.max_vertices and len(features) > self.max_vertices:
                indices = np.random.choice(len(features), self.max_vertices, replace=False)
                features, labels = features[indices], labels[indices]

            if self.augment:
                features = self._augment_features(features)
                data_obj = Data(pos=torch.from_numpy(features[:, :3]))
                distorted_data_obj = self.elastic_distortion(data_obj)
                features[:, :3] = distorted_data_obj.pos.numpy()

            features[:, :3] = self._normalize_pos(features[:, :3])
            arc_params = self._compute_arch_params(features)
            labels = self._remap_labels(labels)

            if len(features) < self.num_points:
                indices = np.random.choice(len(features), self.num_points, replace=True)
            else:
                indices = np.random.choice(len(features), self.num_points, replace=False)

            features, labels, arc_params = features[indices], labels[indices], arc_params[indices]

            return {
                'pos': torch.from_numpy(features[:, :3]).float(),
                'x': torch.from_numpy(features[:, 3:]).float(),
                'y': torch.from_numpy(labels).long(),
                'arc_params': torch.from_numpy(arc_params).float()
            }
        except Exception as e:
            print(f"Error processing sample {idx} ({os.path.basename(obj_path)}): {str(e)}")
            return self.__getitem__((idx + 1) % len(self))

    def get_label_weights(self):
        return self.label_weights
