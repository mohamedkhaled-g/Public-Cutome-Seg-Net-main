import os
import numpy as np
import json
import torch
from torch.utils.data import Dataset
import random
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


class ObjDataset(Dataset):
    def __init__(self, obj_dir, json_dir, num_points=8192, num_superpoints=256,
                 augment=True, max_vertices=100000, binary=False, num_classes=35):
        self.obj_dir = obj_dir
        self.json_dir = json_dir
        self.num_points = num_points
        self.num_superpoints = num_superpoints
        self.augment = augment
        self.max_vertices = max_vertices
        self.binary = binary
        self.num_classes = num_classes

        # Validate and pair obj/json files
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

        # Compute class weights
        self.label_weights = self._compute_label_weights()

    def _compute_label_weights(self):
        label_counts = np.zeros(self.num_classes)
        for json_file in tqdm(self.json_files, desc="Computing label weights"):
            try:
                labels = self._read_ground_truth(json_file)
                labels = self._remap_labels(labels)
                # Ensure labels are within the expected range before counting
                valid_mask = (labels >= 0) & (labels < self.num_classes)
                unique, counts = np.unique(labels[valid_mask], return_counts=True)
                label_counts[unique] += counts
            except Exception as e:
                print(f"Warning: Error processing {json_file} for weight calculation: {str(e)}")
                continue

        # Use log smoothing for stability with rare classes
        label_counts = np.where(label_counts == 0, 1, label_counts)  # Avoid division by zero
        log_counts = np.log(label_counts)
        weights = (log_counts.max() - log_counts) + 1
        weights = weights / weights.sum() * self.num_classes  # Normalize

        print(f"Computed label weights for {self.num_classes} classes.")
        return torch.tensor(weights, dtype=torch.float32)

    def _read_obj_file(self, obj_path):
        vertices = []
        with open(obj_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if parts and parts[0] == 'v':
                    try:
                        vertices.append(list(map(float, parts[1:4])))
                    except (ValueError, IndexError):
                        continue
        vertices = np.array(vertices, dtype=np.float32)
        if len(vertices) == 0:
            raise ValueError(f"No vertices found in {obj_path}")
        return vertices

    def _read_ground_truth(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        labels = np.array(data['labels'], dtype=np.int64)
        # In multi-class, we might have more vertices than labels if instances are not fully labeled
        # We will handle this by taking the minimum length later
        return labels

    def _remap_labels(self, labels):
        remapped = np.zeros_like(labels)
        if self.binary:
            # Gingiva: 0, Teeth: 1
            remapped = np.where(labels == 0, 0, 1)
        else:
            # Multi-class remapping for 35 classes
            # Gingiva: 0, Upper jaw (11-27): 1-17, Lower jaw (31-47): 18-34
            remapped = np.where(labels == 0, 0, remapped)
            # Upper Jaw
            for fdi in range(11, 28):
                remapped = np.where(labels == fdi, fdi - 10, remapped)
            # Lower Jaw
            for fdi in range(31, 48):
                remapped = np.where(labels == fdi, fdi - 30 + 17, remapped)
        return remapped

    def _augment_pointcloud(self, vertices):
        vertices = vertices.copy()
        if random.random() > 0.5:
            angle = np.random.uniform(-np.pi / 18, np.pi / 18)  # +/- 10 degrees
            rot_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                   [np.sin(angle), np.cos(angle), 0],
                                   [0, 0, 1]], dtype=np.float32)
            vertices = vertices @ rot_matrix.T
        if random.random() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            vertices *= scale
        if random.random() > 0.5:
            jitter = np.random.normal(0, 0.005, vertices.shape).astype(np.float32)
            vertices += jitter
        return vertices

    def _normalize_vertices(self, vertices):
        centroid = np.mean(vertices, axis=0)
        vertices = vertices - centroid
        max_dist = np.max(np.sqrt(np.sum(vertices ** 2, axis=1)))
        if max_dist > 1e-8:
            vertices = vertices / max_dist
        return vertices

    def _resample_points(self, vertices, labels):
        num_verts = len(vertices)
        if num_verts < self.num_points:
            indices = np.random.choice(num_verts, self.num_points, replace=True)
        else:
            indices = np.random.choice(num_verts, self.num_points, replace=False)
        return vertices[indices], labels[indices]

    def _compute_superpoints(self, vertices):
        kmeans = MiniBatchKMeans(n_clusters=self.num_superpoints, random_state=42, batch_size=2048, n_init=10)
        cluster_labels = kmeans.fit_predict(vertices)
        cluster_centers = kmeans.cluster_centers_
        return cluster_labels, cluster_centers

    def __len__(self):
        return len(self.obj_files)

    def __getitem__(self, idx):
        obj_path = self.obj_files[idx]
        json_path = self.json_files[idx]

        try:
            vertices = self._read_obj_file(obj_path)
            labels = self._read_ground_truth(json_path)

            # Ensure vertices and labels align, especially if JSON has fewer labels
            min_len = min(len(vertices), len(labels))
            vertices, labels = vertices[:min_len], labels[:min_len]

            if self.max_vertices and len(vertices) > self.max_vertices:
                indices = np.random.choice(len(vertices), self.max_vertices, replace=False)
                vertices, labels = vertices[indices], labels[indices]

            if self.augment:
                vertices = self._augment_pointcloud(vertices)

            vertices = self._normalize_vertices(vertices)
            labels = self._remap_labels(labels)
            vertices, labels = self._resample_points(vertices, labels)

            superpoint_labels, superpoint_centers = self._compute_superpoints(vertices)

            return {
                'vertices': torch.tensor(vertices, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.long),
                'superpoint_labels': torch.tensor(superpoint_labels, dtype=torch.long),
                'superpoint_centers': torch.tensor(superpoint_centers, dtype=torch.float32),
            }
        except Exception as e:
            print(f"Error processing sample {idx} ({os.path.basename(obj_path)}): {str(e)}")
            # Return a dummy sample on error
            return self.__getitem__((idx + 1) % len(self))

    def get_label_weights(self):
        return self.label_weights
