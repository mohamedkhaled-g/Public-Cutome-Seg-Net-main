import os
import numpy as np
import json
import torch
from torch.utils.data import Dataset
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict


class DentalDataset(Dataset):
    def __init__(self, obj_dir, json_dir, num_points=8192, max_vertices=250000,
                 augment=True, preload=False, cache_limit=20):
        self.num_points = num_points
        self.max_vertices = max_vertices
        self.augment = augment
        self.preload = preload
        self.cache_limit = cache_limit * 1024 ** 3
        self.cache = {}
        self.current_cache_size = 0
        self.class_distribution = defaultdict(int)

        # Verify directories exist
        if not os.path.exists(obj_dir):
            raise FileNotFoundError(f"OBJ directory not found: {obj_dir}")
        if not os.path.exists(json_dir):
            raise FileNotFoundError(f"JSON directory not found: {json_dir}")

        # Find and validate file pairs
        self.pairs = self._find_valid_pairs(obj_dir, json_dir)

        if not self.pairs:
            raise ValueError(f"No valid OBJ-JSON pairs found in:\nOBJ dir: {obj_dir}\nJSON dir: {json_dir}")

        print(f"Successfully initialized dataset with {len(self)} samples")

        if self.preload:
            self._preload_data()

        # Analyze initial class distribution
        self._analyze_class_distribution()

    def __len__(self):
        """Returns the number of samples in the dataset"""
        return len(self.pairs)

    def __getitem__(self, idx):
        """Load and return a sample from the dataset"""
        obj_path, json_path, _ = self.pairs[idx]

        # Try to get from cache
        if obj_path in self.cache:
            vertices, labels = self.cache[obj_path]
        else:
            # Load from disk
            vertices = self._load_obj(obj_path)
            labels = self._load_json(json_path)

            # Cache if possible
            if self.current_cache_size + vertices.nbytes + labels.nbytes < self.cache_limit:
                self.cache[obj_path] = (vertices, labels)
                self.current_cache_size += vertices.nbytes + labels.nbytes

        # Data augmentation
        if self.augment:
            vertices = self._augment_data(vertices)

        # Random sampling if needed
        if vertices.shape[0] > self.num_points:
            indices = np.random.choice(vertices.shape[0], self.num_points, replace=False)
            vertices = vertices[indices]
            labels = labels[indices]
        elif vertices.shape[0] < self.num_points:
            # Pad with zeros if needed
            padding = np.zeros((self.num_points - vertices.shape[0], 3))
            vertices = np.vstack([vertices, padding])
            labels = np.pad(labels, (0, self.num_points - labels.shape[0]), 'constant')

        return {
            'vertices': torch.FloatTensor(vertices),
            'labels': torch.LongTensor(labels)
        }

    def _find_valid_pairs(self, obj_dir, json_dir):
        """Find and validate OBJ-JSON pairs with size checking"""
        pairs = []
        print(f"Searching for OBJ files in {obj_dir}...")

        for root, _, files in os.walk(obj_dir):
            for f in files:
                if f.endswith('.obj'):
                    obj_path = os.path.join(root, f)
                    rel_path = os.path.relpath(obj_path, obj_dir)
                    json_path = os.path.join(json_dir, rel_path.replace('.obj', '.json'))

                    if not os.path.exists(json_path):
                        print(f"Warning: No matching JSON found for {obj_path}")
                        continue

                    try:
                        # Quick validation check
                        labels = self._load_json(json_path, validate_only=True)
                        file_size = os.path.getsize(obj_path) + os.path.getsize(json_path)

                        if file_size < self.cache_limit:
                            pairs.append((obj_path, json_path, file_size))
                        else:
                            pairs.append((obj_path, json_path, 0))
                            print(f"File pair too large for caching: {obj_path} ({file_size / 1024 ** 2:.2f}MB)")
                    except Exception as e:
                        print(f"Skipping invalid pair {obj_path}: {str(e)}")

        print(f"Found {len(pairs)} valid OBJ-JSON pairs")
        return pairs

    def _load_obj(self, path):
        """Load vertices from OBJ file"""
        vertices = []
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    vertices.append([float(x) for x in line.strip().split()[1:4]])

        vertices = np.array(vertices, dtype=np.float32)

        if len(vertices) > self.max_vertices:
            vertices = vertices[:self.max_vertices]

        return vertices

    def _load_json(self, path, validate_only=False):
        """Load and convert labels from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)

        labels = np.array(data['labels'], dtype=np.int16)

        if validate_only:
            return labels[:1000]  # Only validate first 1000 labels

        if len(labels) > self.max_vertices:
            labels = labels[:self.max_vertices]

        # Convert to FDI numbering system
        converted = np.zeros_like(labels)
        for i, label in enumerate(labels):
            if label == 0:
                converted[i] = 0  # Gingiva
            elif 11 <= label <= 18:
                converted[i] = label - 10  # Upper right (1-8)
            elif 21 <= label <= 28:
                converted[i] = label - 20 + 8  # Upper left (9-16)
            elif 31 <= label <= 38:
                converted[i] = label - 30 + 16  # Lower left (17-24)
            elif 41 <= label <= 48:
                converted[i] = label - 40 + 24  # Lower right (25-32)
            else:
                converted[i] = 0  # Unknown/invalid

        return converted

    def _augment_data(self, vertices):
        """Apply random augmentations to vertices"""
        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-np.pi / 12, np.pi / 12)
            rot_mat = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            vertices = vertices @ rot_mat

        # Random scaling
        if random.random() > 0.5:
            scale = random.uniform(0.9, 1.1)
            vertices = vertices * scale

        # Random translation
        if random.random() > 0.5:
            translation = np.random.uniform(-0.05, 0.05, 3)
            vertices = vertices + translation

        return vertices

    def _analyze_class_distribution(self):
        """Analyze initial class distribution for weight calculation"""
        sample_size = min(1000, len(self.pairs))
        for _, json_path, _ in random.sample(self.pairs, sample_size):
            labels = self._load_json(json_path)
            unique, counts = np.unique(labels, return_counts=True)
            for cls, cnt in zip(unique, counts):
                self.class_distribution[cls] += cnt

    def get_label_weights(self, power=0.5):
        """
        Compute balanced weights with adjustable power parameter
        power: 0.5 for sqrt weighting, 1.0 for inverse frequency
        """
        counts = np.zeros(35)
        for cls, cnt in self.class_distribution.items():
            counts[cls] = cnt

        # Add smoothing and compute weights
        counts = np.maximum(counts, 1)
        weights = 1.0 / (counts ** power)
        return torch.tensor(weights / weights.sum(), dtype=torch.float32)

    def _preload_data(self):
        """Preload all data into memory"""
        print("Preloading data into memory...")
        for idx in range(len(self)):
            self.__getitem__(idx)  # This will cache all items
        print(f"Preloaded {len(self.cache)}/{len(self)} samples into memory")