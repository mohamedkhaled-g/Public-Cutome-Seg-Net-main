# dataloader/Dataloader_TeethGnn.py
import os
import numpy as np
import torch
from torch_geometric.data import Data
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import json
import trimesh  # A more robust library for mesh I/O
import time


class TeethGNNDataset:
    def __init__(self, obj_dir, json_dir, num_facets=10000, num_classes=53, augment=True):
        self.obj_dir = obj_dir
        self.json_dir = json_dir
        self.num_facets = num_facets
        self.num_classes = num_classes
        self.augment = augment

        # Find all .obj and .json pairs
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
            raise ValueError("No valid .obj/.json pairs found.")
        print(f"Found {len(self.obj_files)} samples.")

    def _remap_labels(self, labels):
        """
        Map FDI two-digit notation to a continuous range.
        num_classes=53 -> permanent + deciduous teeth mapped to [0, 52].
        Otherwise -> legacy 35-class mapping (0-34).
        """
        if self.num_classes != 53:
            labels = np.where(labels == 0, 0, labels)
            labels = np.where((labels >= 11) & (labels <= 28), labels - 10, labels)
            labels = np.where((labels >= 31) & (labels <= 48), labels - 30 + 17, labels)
            return np.clip(labels, 0, 34)

        # 53-class FDI mapping: 0, 1-8, 9-16, 17-24, 25-32, 33-37, 38-42, 43-47, 48-52
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
        return np.clip(remapped, 0, 52)

    def _load_mesh_and_labels(self, obj_path, json_path):
        start_time = time.time()
        # 1. Load the mesh
        mesh = trimesh.load(obj_path, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Failed to load {obj_path} as a triangular mesh.")
        load_mesh_time = time.time() - start_time

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.faces)

        # 2. Load JSON labels (per-vertex)
        start_time = time.time()
        with open(json_path, 'r') as f:
            data = json.load(f)
        vertex_labels = np.array(data['labels'], dtype=np.int64)
        load_labels_time = time.time() - start_time

        # 3. Remap FDI labels to contiguous range [0, num_classes)
        start_time = time.time()
        remapped_labels = self._remap_labels(vertex_labels.copy())
        remap_time = time.time() - start_time

        print(f"  _load_mesh_and_labels completed in {load_mesh_time:.2f}s (load mesh) + "
              f"{load_labels_time:.2f}s (load labels) + {remap_time:.2f}s (remap) = {time.time() - start_time:.2f}s")

        return mesh, remapped_labels

    def _simplify_mesh(self, mesh, target_faces=10000):
        start_time = time.time()
        # Convert to Open3D for simplification
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
        o3d_mesh.compute_triangle_normals()

        # Simplify
        o3d_mesh_simplified = o3d_mesh.simplify_quadric_decimation(target_faces)

        # Convert back to trimesh
        simplified_mesh = trimesh.Trimesh(
            vertices=np.asarray(o3d_mesh_simplified.vertices),
            faces=np.asarray(o3d_mesh_simplified.triangles),
            process=False
        )
        simplify_time = time.time() - start_time
        print(f"  _simplify_mesh completed in {simplify_time:.2f}s")
        return simplified_mesh

    def _extract_dual_graph_features(self, mesh):
        start_time = time.time()
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.faces)
        normals = np.asarray(mesh.face_normals)

        # Compute facet centroids
        centroids = vertices[triangles].mean(axis=1)

        # Compute corner vectors (centroid to each vertex)
        corner_vectors = vertices[triangles] - centroids[:, None, :]
        # Reshape from (N, 3, 3) to (N, 9)
        corner_vectors = corner_vectors.reshape(-1, 9)

        # Node feature X: [centroid (3), normal (3), corner_vectors (9)] -> (N, 15)
        X = np.hstack([centroids, normals, corner_vectors])

        feature_time = time.time() - start_time
        print(f"  _extract_dual_graph_features completed in {feature_time:.2f}s")
        return X, centroids

    def _build_dual_graph_edges(self, centroids, k=16):
        start_time = time.time()
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(centroids)
        distances, indices = nbrs.kneighbors(centroids)

        edge_list = []
        for i, neighbors in enumerate(indices):
            for j in neighbors:
                if i != j:
                    edge_list.append([i, j])
                    edge_list.append([j, i])
        edge_index = np.array(edge_list).T
        edge_build_time = time.time() - start_time
        print(f"  _build_dual_graph_edges completed in {edge_build_time:.2f}s")
        return torch.from_numpy(edge_index).long()

    def __len__(self):
        return len(self.obj_files)

    def __getitem__(self, idx):
        obj_path = self.obj_files[idx]
        json_path = self.json_files[idx]
        print(f"Processing sample {idx}: {obj_path}")

        start_time = time.time()
        try:
            # 1. Load mesh and labels
            mesh, vertex_labels = self._load_mesh_and_labels(obj_path, json_path)

            # 2. Simplify the mesh
            mesh_simplified = self._simplify_mesh(mesh, self.num_facets)
            vertices_simplified = np.asarray(mesh_simplified.vertices)
            triangles_simplified = np.asarray(mesh_simplified.faces)

            # --- Transfer labels ---
            start_transfer_time = time.time()
            nbrs = NearestNeighbors(n_neighbors=1).fit(np.asarray(mesh.vertices))
            _, indices = nbrs.kneighbors(vertices_simplified)
            indices = indices.flatten()

            vertex_labels_simplified = vertex_labels[indices]
            facet_labels_simplified = vertex_labels_simplified[triangles_simplified[:, 0]]
            transfer_time = time.time() - start_transfer_time
            print(f"  Label transfer completed in {transfer_time:.2f}s")

            # 3. Extract features
            X, centroids = self._extract_dual_graph_features(mesh_simplified)
            P = centroids

            # 4. Build dual graph edges
            edge_index = self._build_dual_graph_edges(centroids)

            # 5. Create PyG Data object
            data = Data(
                x=torch.tensor(X, dtype=torch.float),
                pos=torch.tensor(P, dtype=torch.float),
                edge_index=edge_index,
                y=torch.tensor(facet_labels_simplified, dtype=torch.long)
            )

            total_time = time.time() - start_time
            print(f"  __getitem__ completed for sample {idx} in {total_time:.2f}s\n")
            return data

        except Exception as e:
            print(f"Error processing {obj_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))