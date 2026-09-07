# dataloader/DataLoader_kpconv.py
import os

import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset

from dataloader.PreprocessingPaper import (
    read_obj_features,
    read_ground_truth,
    preprocess_sample,
    paper_class_weights,
    PAPER_NUM_POINTS,
    PAPER_MAX_VERTICES,
)


def read_obj_file(obj_path):
    return read_obj_features(obj_path)


class ObjDataset(Dataset):
    """
    A PyTorch Dataset for .obj point clouds + JSON ground truth.
    Follows the paper preprocessing (Algorithm 1): normalise, +/-10 deg Z-rot,
    scale [0.9, 1.1], ElasticDistortion (training only), 65,536-point sampling.
    Returns a PyG Data with pos = xyz and x = 6-D features (xyz + rgb).
    """

    def __init__(self, obj_dir, json_dir, num_points=PAPER_NUM_POINTS,
                 transform=None, num_classes=53, augment=True,
                 max_vertices=PAPER_MAX_VERTICES):
        self.obj_dir = obj_dir
        self.json_dir = json_dir
        self.num_points = num_points
        self.transform = transform
        self.num_classes = num_classes
        self.augment = augment
        self.max_vertices = max_vertices

        self.obj_files = []
        for root, _, files in os.walk(obj_dir):
            for file in files:
                if file.endswith('.obj'):
                    self.obj_files.append(os.path.join(root, file))

        self.json_files = []
        for obj_file in self.obj_files:
            relative_path = os.path.relpath(obj_file, obj_dir)
            json_file = os.path.join(json_dir, relative_path.replace('.obj', '.json'))
            if os.path.exists(json_file):
                self.json_files.append(json_file)
            else:
                raise FileNotFoundError(
                    f"Corresponding JSON file not found for {obj_file}. Expected: {json_file}")

        print(f"Found {len(self.obj_files)} .obj files in {obj_dir}")
        print(f"Found {len(self.json_files)} .json files in {json_dir}")

    def get_label_weights(self):
        """Paper Eq. 2 inverse-frequency weights (1/log(1.02 + |C_c|)), sum == num_classes."""
        weights = paper_class_weights(self.json_files, self.num_classes)
        return torch.from_numpy(weights)

    def __len__(self):
        return len(self.obj_files)

    def __getitem__(self, idx):
        obj_path = self.obj_files[idx]
        json_path = self.json_files[idx]

        features = read_obj_features(obj_path)
        labels = read_ground_truth(json_path)

        features, labels = preprocess_sample(
            features, labels,
            num_points=self.num_points,
            augment=self.augment,
            max_vertices=self.max_vertices,
        )

        vertices = torch.from_numpy(features.astype(np.float32))
        labels = torch.from_numpy(labels.astype(np.int64))

        data = Data(pos=vertices[:, :3], x=vertices, y=labels)

        if self.transform:
            data = self.transform(data)

        return data