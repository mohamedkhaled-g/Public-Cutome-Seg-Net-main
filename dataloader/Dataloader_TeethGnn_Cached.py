# dataloader/Dataloader_TeethGnn_Cached.py
import os
import random

import numpy as np
import torch
from torch_geometric.data import Data, Dataset


class TeethGNNDatasetCached(Dataset):
    """Mesh dual-graph dataset cached as .pt files.

    Paper preprocessing (Algorithm 1) applied at training time only
    (validation uses augment=False): random Z-rotation within +/-10 deg and
    scale in [0.9, 1.1]. The rotation/scale is applied coherently to the
    vertex positions (data.pos = facet centroids) and to the nodal feature
    columns derived from geometry ([centroid(3), normal(3), corner_vectors(9)]).
    """

    def __init__(self, processed_dir, num_classes=53, augment=False):
        super().__init__()

        self._processed_dir = processed_dir
        self._num_classes = num_classes
        self._augment = augment

        self.file_list = [f for f in os.listdir(self._processed_dir) if f.endswith('.pt')]
        self.file_list.sort()

        if not self.file_list:
            raise ValueError(f"No .pt files found in {self._processed_dir}")

        print(f"Found {len(self.file_list)} pre-processed samples (augment={self._augment}).")

    def len(self):
        return len(self.file_list)

    def _augment_data(self, data):
        pos = data.pos.numpy().astype(np.float64)
        angle = np.random.uniform(-np.pi / 18, np.pi / 18) if random.random() > 0.5 else 0.0
        scale = np.random.uniform(0.9, 1.1) if random.random() > 0.5 else 1.0

        rot = np.eye(3)
        if angle != 0.0:
            rot = np.array([[np.cos(angle), -np.sin(angle), 0],
                            [np.sin(angle), np.cos(angle), 0],
                            [0, 0, 1]], dtype=np.float64)

        pos = pos @ rot.T
        if scale != 1.0:
            pos *= scale

        x = data.x.numpy().astype(np.float64)
        # x layout: [centroid(3), normal(3), corner_vectors(9)] = 15 channels
        if x.shape[1] >= 6:
            if angle != 0.0:
                x[:, :3] = x[:, :3] @ rot.T   # centroids
                x[:, 3:6] = x[:, 3:6] @ rot.T # normals
            if scale != 1.0:
                x[:, :3] *= scale            # centroids
        if x.shape[1] >= 15 and (angle != 0.0 or scale != 1.0):
            corner = x[:, 6:15].reshape(-1, 3, 3)
            if angle != 0.0:
                corner = corner @ rot.T
            if scale != 1.0:
                corner *= scale
            x[:, 6:15] = corner.reshape(-1, 9)

        edges = data.edge_index
        y = data.y
        return Data(pos=torch.from_numpy(pos.astype(np.float32)),
                    x=torch.from_numpy(x.astype(np.float32)),
                    edge_index=edges,
                    y=y)

    def get(self, idx):
        filepath = os.path.join(self._processed_dir, self.file_list[idx])
        data = torch.load(filepath)
        if self._augment:
            data = self._augment_data(data)
        return data

    def get_label_weights(self, max_ratio=20.0):
        class_counts = torch.zeros(self._num_classes, dtype=torch.float64)
        for i in range(len(self.file_list)):
            data = self.get(i)
            y = data.y.flatten()
            ids, counts = y.unique(return_counts=True)
            valid = ids < self._num_classes
            ids, counts = ids[valid], counts[valid]
            class_counts[ids] += counts.float()

        total = class_counts.sum()
        if total == 0:
            return torch.ones(self._num_classes, dtype=torch.float32)

        weights = total / (self._num_classes * (class_counts + 1.0))
        weights = weights / weights.mean()
        weights = weights.clamp(1.0 / max_ratio, max_ratio)
        return weights.float()