# dataloader/DataLoader_HybridTransformer.py

import os
import numpy as np
import json
import torch
from torch.utils.data import Dataset
import random
from tqdm import tqdm
# You may need to install this library: pip install torch-points3d --no-deps  # then manually install dependencies
from torch_points3d.core.data_transform import ElasticDistortion
from torch_geometric.data import Data


class ObjDataset(Dataset):
    def __init__(self, obj_dir, json_dir, num_points=8192, augment=True,
                 max_vertices=100000, binary=False, num_classes=53): # Change default to 53
        self.obj_dir = obj_dir
        self.json_dir = json_dir
        self.num_points = num_points
        self.augment = augment
        self.max_vertices = max_vertices
        self.binary = binary
        self.num_classes = num_classes # Should be 53 for FDI

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
                labels = self._remap_labels(labels) # This will now map FDI to 0-52
                # Ensure remapped labels are within the expected range [0, num_classes)
                valid_mask = (labels >= 0) & (labels < self.num_classes)
                if not valid_mask.all():
                    print(f"Warning: Found labels outside [0, {self.num_classes}) range after remapping in {json_file}. Check _remap_labels logic.")
                unique, counts = np.unique(labels[valid_mask], return_counts=True)
                label_counts[unique] += counts
            except Exception as e:
                print(f"Warning: Error processing {json_file} for weight calculation: {str(e)}")

        label_counts = np.where(label_counts == 0, 1, label_counts) # Avoid log(0)
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
        """
        Remap FDI two-digit notation to a continuous range [0, 52].
        FDI codes: 0, 11-18, 21-28, 31-38, 41-48, 51-55, 61-65, 71-75, 81-85
        Mapped indices: 0, 1-8, 9-16, 17-24, 25-32, 33-37, 38-42, 43-47, 48-52
        """
        remapped = np.full_like(labels, -1) # Initialize with invalid value

        # Map background (0) to index 0
        remapped = np.where(labels == 0, 0, remapped)

        # Map Permanent Teeth
        # Upper Right (18-11) -> indices 1-8
        for fdi in range(11, 19): # 11 to 18 inclusive
            remapped = np.where(labels == fdi, fdi - 10, remapped) # 11->1, 12->2, ..., 18->8
        # Upper Left (21-28) -> indices 9-16
        for fdi in range(21, 29): # 21 to 28 inclusive
            remapped = np.where(labels == fdi, fdi - 20 + 8, remapped) # 21->9, 22->10, ..., 28->16
        # Lower Left (31-38) -> indices 17-24
        for fdi in range(31, 39): # 31 to 38 inclusive
            remapped = np.where(labels == fdi, fdi - 30 + 16, remapped) # 31->17, 32->18, ..., 38->24
        # Lower Right (41-48) -> indices 25-32
        for fdi in range(41, 49): # 41 to 48 inclusive
            remapped = np.where(labels == fdi, fdi - 40 + 24, remapped) # 41->25, 42->26, ..., 48->32

        # Map Deciduous Teeth
        # Upper Right (55-51) -> indices 33-37
        for fdi in range(51, 56): # 51 to 55 inclusive
            remapped = np.where(labels == fdi, fdi - 50 + 32, remapped) # 51->33, 52->34, ..., 55->37
        # Upper Left (61-65) -> indices 38-42
        for fdi in range(61, 66): # 61 to 65 inclusive
            remapped = np.where(labels == fdi, fdi - 60 + 37, remapped) # 61->38, 62->39, ..., 65->42
        # Lower Left (71-75) -> indices 43-47
        for fdi in range(71, 76): # 71 to 75 inclusive
            remapped = np.where(labels == fdi, fdi - 70 + 42, remapped) # 71->43, 72->44, ..., 75->47
        # Lower Right (81-85) -> indices 48-52
        for fdi in range(81, 86): # 81 to 85 inclusive
            remapped = np.where(labels == fdi, fdi - 80 + 47, remapped) # 81->48, 82->49, ..., 85->52

        # Check for unmapped labels (should ideally be none if all FDI codes are handled)
        unmapped_mask = remapped == -1
        if unmapped_mask.any():
             print(f"Warning: Found {unmapped_mask.sum()} labels that could not be remapped. Unique values: {np.unique(labels[unmapped_mask])}")

        return remapped


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
            labels = self._remap_labels(labels) # Apply the new remapping

            if len(features) < self.num_points:
                indices = np.random.choice(len(features), self.num_points, replace=True)
            else:
                indices = np.random.choice(len(features), self.num_points, replace=False)

            features, labels = features[indices], labels[indices]

            return {
                'pos': torch.from_numpy(features[:, :3]).float(),
                'x': torch.from_numpy(features[:, 3:]).float(),
                'y': torch.from_numpy(labels).long() # This should now be in range [0, 52]
            }
        except Exception as e:
            print(f"Error processing sample {idx} ({os.path.basename(obj_path)}): {str(e)}")
            return self.__getitem__((idx + 1) % len(self))

    def get_label_weights(self):
        return self.label_weights
