# networks/TeethGnn_Model.py
import torch
import torch.nn as nn
import torch.nn.functional as FF
from torch_geometric.nn import EdgeConv, global_max_pool
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN


def MLP(channels, batch_norm=True):
    layers = []
    for i in range(1, len(channels)):
        layers.append(Lin(channels[i - 1], channels[i]))
        if batch_norm and i < len(channels) - 1:
            layers.append(BN(channels[i]))
        if i < len(channels) - 1:
            layers.append(ReLU())
    return Seq(*layers)


class StaticEdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nn = MLP([in_channels * 2, out_channels])
        self.edge_conv = EdgeConv(self.nn)

    def forward(self, x, edge_index):
        return self.edge_conv(x, edge_index)


class TeethGNN(nn.Module):
    def __init__(self, num_classes=35, k=16):
        super().__init__()
        self.k = k

        # Feature Extractor (Modified DGCNN)
        self.conv1 = StaticEdgeConv(15, 64)
        self.conv2 = StaticEdgeConv(64, 64)
        self.conv3 = StaticEdgeConv(64, 128)

        # Global feature aggregation
        self.fc_global = MLP([128, 1024], batch_norm=False)

        # Final node feature F
        self.fc_f = Lin(128 + 1024, 1216)

        # Two-Head Predictor
        self.semantic_head = MLP([1216, 512, num_classes])
        self.offset_head = MLP([1216, 256, 3])  # Output 3D offset vector

    def forward(self, data):
        x, edge_index, pos = data.x, data.edge_index, data.pos

        # Feature Extractor
        x1 = FF.relu(self.conv1(x, edge_index))
        x2 = FF.relu(self.conv2(x1, edge_index))
        x3 = FF.relu(self.conv3(x2, edge_index))  # (N, 128)

        # Global feature
        global_feat = global_max_pool(x3, batch=None)
        global_feat = self.fc_global(global_feat).repeat(x3.size(0), 1)  # (N, 1024)

        # Node-wise feature F
        F = torch.cat([x3, global_feat], dim=1)  # (N, 1152)
        F = self.fc_f(F)  # (N, 1216)

        # Two-Head Prediction
        semantic_logits = self.semantic_head(F)  # (N, 35)
        offsets = self.offset_head(F)  # (N, 3)

        return semantic_logits, offsets, pos