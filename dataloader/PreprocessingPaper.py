# dataloader/PreprocessingPaper.py
# Shared preprocessing for ALL models, matching Dental_Springer_BMC_Adjusted_v3.pdf
# Algorithm 1 / Section 6.1:
#   - read 6-D vertex features (xyz + rgb)
#   - random Z-rotation within +/-10 deg (pi/18), scale in [0.9, 1.1], color jitter
#   - ElasticDistortion (training only)
#   - centre on centroid and unit-vector normalisation (max distance = 1)
#   - FDI -> [0, 52] label remapping (53 classes)
#   - sample to 65,536 points
#   - class weights w_c = (1 / log(1.02 + |C_c|)) / sum(...) * num_classes  (Eq. 2)
import json
import os
import random

import numpy as np
import torch
from torch_geometric.data import Data

from torch_points3d.core.data_transform import ElasticDistortion

PAPER_NUM_POINTS = 65536
PAPER_ROT_MAX = np.pi / 18.0      # +/- 10 degrees
PAPER_SCALE_MIN = 0.9
PAPER_SCALE_MAX = 1.1
PAPER_MAX_VERTICES = 100000       # cap before the N=65,536 sampling (same setting as the Hybrid pipeline)
PAPER_TRAIN_RATIO = 0.7
PAPER_VAL_RATIO = 0.2
PAPER_TEST_RATIO = 0.1


def read_obj_features(obj_path):
    """Return (N, 6) float32 features [xyz, rgb]. Files carrying only xyz get rgb = 0."""
    rows = []
    with open(obj_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts and parts[0] == 'v':
                try:
                    values = list(map(float, parts[1:7]))
                except (ValueError, IndexError):
                    continue
                if len(values) == 6:
                    rows.append(values)
    features = np.asarray(rows, dtype=np.float32)
    if len(features) == 0:
        raise ValueError(f"No 6-column vertices found in {obj_path}")
    return features


def read_ground_truth(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return np.asarray(data['labels'], dtype=np.int64)


def remap_labels_fdi(labels, num_classes=53):
    """FDI two-digit notation -> [0, 52] (53 classes). Same mapping as the Hybrid pipeline."""
    if num_classes != 53:
        remapped = np.where(labels == 0, 0, labels)
        remapped = np.where((labels >= 11) & (labels <= 28), labels - 10, remapped)
        remapped = np.where((labels >= 31) & (labels <= 48), labels - 30 + 17, remapped)
        return np.clip(remapped, 0, num_classes - 1)

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

    unmapped = remapped < 0
    if unmapped.any():
        print(f"Warning: {unmapped.sum()} labels could not be remapped. "
              f"Unique: {np.unique(labels[unmapped])}")
    return np.clip(remapped, 0, num_classes - 1)


def augment_features(features):
    """Random Z-rotation +/-10 deg, scale [0.9, 1.1], color jitter +/-(+-0.1)."""
    pos = features[:, :3].copy()
    if random.random() > 0.5:
        angle = np.random.uniform(-PAPER_ROT_MAX, PAPER_ROT_MAX)
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                               [np.sin(angle), np.cos(angle), 0],
                               [0, 0, 1]], dtype=np.float32)
        pos = pos @ rot_matrix.T
    if random.random() > 0.5:
        pos *= np.random.uniform(PAPER_SCALE_MIN, PAPER_SCALE_MAX)

    colors = features[:, 3:].copy()
    if random.random() > 0.5:
        colors += np.random.uniform(-0.1, 0.1, 3)
    colors = np.clip(colors, 0, 1)

    return np.hstack([pos, colors])


def elastic_distort(features, elastic_distortion=None):
    """ElasticDistortion applied to the point positions (training only)."""
    from torch_points3d.core.data_transform import ElasticDistortion
    if elastic_distortion is None:
        elastic_distortion = ElasticDistortion(apply_distorsion=True,
                                               granularity=[0.2, 0.8],
                                               magnitude=[0.4, 1.6])
    data_obj = Data(pos=torch.from_numpy(features[:, :3]))
    distorted = elastic_distortion(data_obj)
    features[:, :3] = distorted.pos.numpy()
    return features


def normalize_pos(pos):
    """Centroid-centre and unit-radius normalisation (max distance = 1)."""
    pos = pos - np.mean(pos, axis=0)
    max_dist = np.max(np.sqrt(np.sum(pos ** 2, axis=1)))
    if max_dist > 1e-8:
        pos = pos / max_dist
    return pos


def sample_fixed(features, labels, num_points):
    if len(features) < num_points:
        indices = np.random.choice(len(features), num_points, replace=True)
    else:
        indices = np.random.choice(len(features), num_points, replace=False)
    return features[indices], labels[indices]


def preprocess_sample(features, labels, num_points=PAPER_NUM_POINTS,
                      augment=True, max_vertices=PAPER_MAX_VERTICES,
                      elastic_distortion=None):
    """Full Algorithm-1 preprocessing for one sample (same order as the Hybrid pipeline)."""
    min_len = min(len(features), len(labels))
    features, labels = features[:min_len], labels[:min_len]

    if max_vertices and len(features) > max_vertices:
        indices = np.random.choice(len(features), max_vertices, replace=False)
        features, labels = features[indices], labels[indices]

    if augment:
        features = augment_features(features)
        features = elastic_distort(features, elastic_distortion)

    features[:, :3] = normalize_pos(features[:, :3])
    labels = remap_labels_fdi(labels)

    features, labels = sample_fixed(features, labels, num_points)
    return features, labels


def paper_class_weights(json_files, num_classes=53):
    """Eq. 2: w_c = (1 / log(1.02 + |C_c|)) / sum_c(...) * num_classes."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for json_path in json_files:
        labels = remap_labels_fdi(read_ground_truth(json_path), num_classes)
        for c in range(num_classes):
            counts[c] += int(np.sum(labels == c))
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / np.log(1.02 + counts)
    weights = weights / np.sum(weights) * num_classes
    return weights.astype(np.float32)


def paper_split(n_samples, train_ratio=PAPER_TRAIN_RATIO,
                val_ratio=PAPER_VAL_RATIO, test_ratio=PAPER_TEST_RATIO,
                seed=42):
    """70 / 20 / 10 deterministic split. Returns (train_idx, val_idx, test_idx)."""
    rng = np.random.RandomState(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val
    return (indices[:n_train], indices[n_train:n_train + n_val],
            indices[n_train + n_val:])