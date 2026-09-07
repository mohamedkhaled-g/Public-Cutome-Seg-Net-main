import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def knn(x, k):
    """Fixed KNN implementation with proper batch handling"""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def fps(x, n_samples):
    """Farthest Point Sampling with proper batch handling"""
    device = x.device
    B, C, N = x.shape
    centroids = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    for i in range(n_samples):
        centroids[:, i] = farthest
        centroid = x[torch.arange(B), :, farthest].view(B, C, 1)
        dist = torch.sum((x - centroid) ** 2, dim=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


class DilatedEdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k_local=20, k_dilated=100):
        super().__init__()
        self.k_local = k_local
        self.k_dilated = k_dilated

        # Local feature extraction
        self.conv_local = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Dilated feature extraction
        if k_dilated > 0:
            self.conv_dilated = nn.Sequential(
                nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )

        # Residual connection
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        B, C, N = x.shape

        # Local neighborhood features
        idx = knn(x, self.k_local)  # (B, N, k)
        x_expanded = x.unsqueeze(3).expand(-1, -1, -1, self.k_local)
        neighbors = torch.gather(x_expanded, 2, idx.unsqueeze(1).expand(-1, C, -1, -1))
        x_local = torch.cat([x_expanded, neighbors - x_expanded], dim=1)
        x_local = self.conv_local(x_local)
        x_local = torch.max(x_local, dim=3)[0]

        # Residual connection
        res = self.residual(x)

        # Dilated neighborhood features
        if self.k_dilated > 0 and N > self.k_dilated:
            fps_idx = fps(x, self.k_dilated)  # (B, k_dilated)
            fps_points = torch.gather(x, 2, fps_idx.unsqueeze(1).expand(-1, C, -1))

            idx_dilated = knn(fps_points, min(self.k_local, self.k_dilated))
            fps_expanded = fps_points.unsqueeze(3).expand(-1, -1, -1, self.k_local)
            dilated_neighbors = torch.gather(fps_expanded, 2, idx_dilated.unsqueeze(1).expand(-1, C, -1, -1))

            x_dilated = torch.cat([fps_expanded, dilated_neighbors - fps_expanded], dim=1)
            x_dilated = self.conv_dilated(x_dilated)
            x_dilated = torch.max(x_dilated, dim=3)[0]
            x_dilated = F.interpolate(x_dilated, size=N, mode='nearest')
            x_local = x_local + x_dilated  # Feature fusion

        return F.leaky_relu(x_local + res, 0.2)


class DGCNN_Dilated(nn.Module):
    def __init__(self, num_classes=53, k_local=20, k_dilated=[100, 200, 400]):
        super().__init__()
        # Feature extraction layers
        self.edge_conv1 = DilatedEdgeConv(6, 128, k_local, k_dilated[0])
        self.edge_conv2 = DilatedEdgeConv(128, 256, k_local, k_dilated[1])
        self.edge_conv3 = DilatedEdgeConv(256, 512, k_local, k_dilated[2])

        # Improved classifier head
        self.mlp = nn.Sequential(
            nn.Conv1d(128 + 256 + 512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Conv1d(512, num_classes, 1)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, 6, N)

        # Feature extraction
        x1 = self.edge_conv1(x)
        x2 = self.edge_conv2(x1)
        x3 = self.edge_conv3(x2)

        # Feature fusion
        x = torch.cat([x1, x2, x3], dim=1)

        # Classification
        x = self.mlp(x)
        return x.transpose(1, 2)  # (B, N, num_classes)