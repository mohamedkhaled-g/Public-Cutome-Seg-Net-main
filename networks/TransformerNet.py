import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from math import sqrt


class PositionAwareAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = 1 / sqrt(dim_head)
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, dim_head),  # Changed output dimension to match dim_head
            nn.GELU(),
            nn.Linear(dim_head, dim_head)
        )
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, pos):
        B, N, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # Enhanced position encoding
        pos_enc = self.pos_mlp(pos)  # [B, N, dim_head]
        pos_enc = rearrange(pos_enc, 'b n d -> b 1 n d')  # Simplified expansion

        # Position-aware attention
        q = q + pos_enc
        k = k + pos_enc

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        out = torch.matmul(attn, v)

        return self.to_out(rearrange(out, 'b h n d -> b n (h d)'))


class PointTransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, dropout=0.1, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PositionAwareAttention(dim, heads, dim_head, dropout)
        self.norm2 = nn.LayerNorm(dim)

        mlp_dim = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, pos):
        x = x + self.attn(self.norm1(x), pos)
        x = x + self.mlp(self.norm2(x))
        return x


class EfficientPointTransformer(nn.Module):
    def __init__(self, num_classes=35, dim=128, depth=4, heads=4, dim_head=32, dropout=0.1):
        super().__init__()

        # Hierarchical feature extraction
        self.input_proj = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, dim, 1),
            nn.BatchNorm1d(dim),
            nn.GELU()
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            PointTransformerBlock(dim, heads, dim_head, dropout)
            for _ in range(depth)
        ])

        # Adaptive output head
        self.head = nn.Sequential(
            nn.Conv1d(dim, dim // 2, 1),
            nn.BatchNorm1d(dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim // 2, num_classes, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        pos = x  # [B, N, 3]
        x = x.transpose(2, 1)  # [B, 3, N]
        x = self.input_proj(x).transpose(2, 1)  # [B, N, C]

        for block in self.blocks:
            x = block(x, pos)

        return self.head(x.transpose(2, 1)).transpose(2, 1)  # [B, N, num_classes]


class AdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1, adaptive=True):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.adaptive = adaptive

        if alpha is None:
            self.register_buffer('alpha', torch.ones(35))
        else:
            self.register_buffer('alpha', alpha)

        if adaptive:
            self.gamma = nn.Parameter(torch.tensor(gamma))
            self.label_smoothing = nn.Parameter(torch.tensor(label_smoothing))

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        targets = targets.long()

        # Adaptive label smoothing
        if self.adaptive:
            smoothing = torch.sigmoid(self.label_smoothing) * 0.2
            one_hot = torch.zeros_like(inputs).scatter_(-1, targets.unsqueeze(-1), 1)
            one_hot = one_hot * (1 - smoothing) + smoothing / inputs.size(-1)
            loss = -(one_hot * log_probs).sum(dim=-1)
        else:
            loss = F.nll_loss(log_probs, targets, reduction='none')

        # Adaptive focal modulation
        pt = torch.exp(-loss)
        gamma = torch.sigmoid(self.gamma) * 3.0 if self.adaptive else self.gamma
        focal_loss = (1 - pt) ** gamma * loss

        # Apply class weights
        weights = self.alpha[targets]
        return (weights * focal_loss).mean()