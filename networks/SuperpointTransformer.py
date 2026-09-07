import torch
import torch.nn as nn


# This is the final corrected network file.

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class SuperpointTransformer(nn.Module):
    def __init__(self, num_classes=35, num_superpoints=256, nhead=8, d_model=256):
        super().__init__()
        self.num_superpoints = num_superpoints
        self.d_model = d_model
        # --- FIX 1: Save num_classes as an attribute ---
        self.num_classes = num_classes

        self.superpoint_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True)
        )

        self.pos_encoding = nn.Parameter(torch.randn(1, num_superpoints, d_model))

        self.transformer_blocks = nn.Sequential(
            TransformerBlock(d_model, nhead),
            TransformerBlock(d_model, nhead),
            TransformerBlock(d_model, nhead)
        )

        self.seg_head = nn.Sequential(
            nn.Linear(d_model + 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, vertices, superpoint_labels, superpoint_centers):
        B, N, _ = vertices.shape
        _, M, _ = superpoint_centers.shape

        sp_flat = superpoint_centers.reshape(B * M, 3)
        sp_features_flat = self.superpoint_encoder(sp_flat)
        sp_features = sp_features_flat.reshape(B, M, self.d_model)

        sp_features = sp_features + self.pos_encoding
        sp_features = self.transformer_blocks(sp_features)

        sp_indices = superpoint_labels.unsqueeze(-1).expand(-1, -1, self.d_model)
        point_sp_features = torch.gather(sp_features, 1, sp_indices)

        combined_features = torch.cat([point_sp_features, vertices], dim=-1)

        combined_features_flat = combined_features.reshape(B * N, self.d_model + 3)
        logits_flat = self.seg_head(combined_features_flat)

        # --- FIX 2: Use the saved attribute self.num_classes ---
        logits = logits_flat.reshape(B, N, self.num_classes)

        return logits
