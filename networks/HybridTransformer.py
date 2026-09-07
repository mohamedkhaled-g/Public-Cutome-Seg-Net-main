import torch
import torch.nn as nn
import torch.nn.functional as F
from .HybridTransformerPointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ffn(x)
        x = self.norm2(x + ff_output)
        return x

class HybridPointTransformer(nn.Module):
    def __init__(self, num_classes=35, d_model=256, nhead=4, debug=False):
        super().__init__()
        self.debug = debug

        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.1, nsample=32, in_channel=3 + 6, mlp=[64, 64, 128],
                                          group_all=False, debug=self.debug)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=0.2, nsample=32, in_channel=128 + 3, mlp=[128, 128, 256],
                                          group_all=False, debug=self.debug)
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=0.4, nsample=32, in_channel=256 + 3,
                                          mlp=[256, 256, d_model], group_all=False, debug=self.debug)

        self.transformer_blocks = nn.Sequential(
            TransformerBlock(d_model, nhead, dropout=0.5),
            TransformerBlock(d_model, nhead, dropout=0.5),
            TransformerBlock(d_model, nhead, dropout=0.5)
        )

        self.fp3 = PointNetFeaturePropagation(in_channel=d_model + 256, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + 3, mlp=[128, 128, 128])

        self.seg_head = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, pos, x):
        
        
        l0_xyz = pos.permute(0, 2, 1)
        l0_points = torch.cat([pos, x], dim=-1).permute(0, 2, 1)
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        transformer_in = l3_points.permute(0, 2, 1)
        
        transformer_out = self.transformer_blocks(transformer_in)
        
        l3_points = transformer_out.permute(0, 2, 1)

        
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        
        
        
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_xyz, l1_points)
        
        
        logits = self.seg_head(l0_points)
        
        return logits.permute(0, 2, 1)
