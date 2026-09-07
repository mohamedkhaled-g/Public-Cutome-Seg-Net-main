import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def furthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    if N <= npoint:
        return torch.arange(N, device=device).unsqueeze(0).expand(B, -1).contiguous()
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def three_nn_interpolate(xyz1, xyz2, points2, chunk_size=512):
    B, N1, _ = xyz1.shape
    _, N2, _ = xyz2.shape
    if N2 == 1:
        return points2.repeat(1, N1, 1)
    if N1 * N2 <= chunk_size * chunk_size:
        return _three_nn_interpolate_dense(xyz1, xyz2, points2)
    result = []
    for start in range(0, N1, chunk_size):
        end = min(start + chunk_size, N1)
        chunk = _three_nn_interpolate_dense(xyz1[:, start:end], xyz2, points2)
        result.append(chunk)
    return torch.cat(result, dim=1)


def _three_nn_interpolate_dense(xyz1, xyz2, points2):
    B, N1, _ = xyz1.shape
    dists = torch.sum((xyz1.unsqueeze(2) - xyz2.unsqueeze(1)) ** 2, -1)
    dists, idx = dists.sort(dim=-1)
    dists, idx = dists[:, :, :3], idx[:, :, :3]
    dist_recip = 1.0 / (dists + 1e-8)
    norm = torch.sum(dist_recip, dim=-1, keepdim=True)
    weight = dist_recip / norm
    interpolated = torch.sum(index_points(points2, idx) * weight.unsqueeze(-1), dim=-2)
    return interpolated


def gather_neighbors(x, knn_idx):
    B, N, C = x.shape
    k = knn_idx.shape[-1]
    device = x.device
    batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, k)
    return x[batch_indices, knn_idx, :]


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep_prob = 1 - self.drop_prob
        mask = x.new_empty(x.shape[0], 1, 1).bernoulli_(keep_prob)
        return x * mask / keep_prob


class ManifoldVecAttn(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.pos_mlp = nn.Sequential(
            nn.Linear(4, self.head_dim),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(self.head_dim, self.head_dim)
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(self.head_dim * 2, self.head_dim),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(self.head_dim, self.head_dim)
        )
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, pos, arc_params, knn_idx):
        B, N, _ = x.shape
        n_neighbors = knn_idx.shape[-1]

        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q = qkv[:, :, 0].contiguous()
        k = qkv[:, :, 1].contiguous()
        v = qkv[:, :, 2].contiguous()

        q = q * self.scale

        k_neighbors = gather_neighbors(k.reshape(B, N, self.n_heads * self.head_dim), knn_idx)
        v_neighbors = gather_neighbors(v.reshape(B, N, self.n_heads * self.head_dim), knn_idx)
        k_neighbors = k_neighbors.reshape(B, N, n_neighbors, self.n_heads, self.head_dim)
        v_neighbors = v_neighbors.reshape(B, N, n_neighbors, self.n_heads, self.head_dim)

        pos_n = gather_neighbors(pos, knn_idx)
        delta_xyz = pos.unsqueeze(2) - pos_n

        arc_n = gather_neighbors(arc_params, knn_idx)
        delta_arc = arc_params.unsqueeze(2) - arc_n

        pos_enc_input = torch.cat([delta_xyz, delta_arc], dim=-1)
        pos_enc = self.pos_mlp(pos_enc_input)
        pos_enc = pos_enc.unsqueeze(-2).expand(-1, -1, -1, self.n_heads, -1)

        q_expanded = q.unsqueeze(2).expand(-1, -1, n_neighbors, -1, -1)
        attn_input = torch.cat([q_expanded - k_neighbors, pos_enc], dim=-1)
        attn_bias = self.attn_mlp(attn_input)
        attn_weights = F.softmax(attn_bias, dim=-3)

        out = (attn_weights * v_neighbors).sum(dim=-3)
        out = out.reshape(B, N, self.d_model)
        return self.proj(out)


class ManifoldVecAttnBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = ManifoldVecAttn(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, pos, arc_params, knn_idx):
        x = x + self.drop_path(self.attn(self.norm1(x), pos, arc_params, knn_idx))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class HybridKNN(nn.Module):
    def __init__(self, k, alpha_init=0.7, chunk_size=512):
        super().__init__()
        self.k = k
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.chunk_size = chunk_size

    def forward(self, xyz, arc_params):
        B, N, _ = xyz.shape
        if N <= self.chunk_size:
            return self._compute_knn_dense(xyz, arc_params)
        return self._compute_knn_chunked(xyz, arc_params)

    @staticmethod
    def _compute_hybrid_distance(xyz_q, xyz, arc_q, arc, alpha):
        sq_dist = torch.sum((xyz_q.unsqueeze(-2) - xyz.unsqueeze(-3)) ** 2, -1)
        euclidean_dist = torch.sqrt(sq_dist + 1e-8)
        arc_diff = torch.abs(arc_q.unsqueeze(-2) - arc.unsqueeze(-3))
        arc_dist = arc_diff.squeeze(-1)
        return alpha * euclidean_dist + (1 - alpha) * arc_dist

    def _compute_knn_dense(self, xyz, arc_params):
        alpha_clamped = torch.sigmoid(self.alpha)
        hybrid_dist = self._compute_hybrid_distance(xyz, xyz, arc_params, arc_params, alpha_clamped)
        _, knn_idx = hybrid_dist.topk(self.k, dim=-1, largest=False)
        return knn_idx

    def _compute_knn_chunked(self, xyz, arc_params):
        B, N, _ = xyz.shape
        alpha_clamped = torch.sigmoid(self.alpha)
        knn_idx = torch.zeros(B, N, self.k, dtype=torch.long, device=xyz.device)
        for start in range(0, N, self.chunk_size):
            end = min(start + self.chunk_size, N)
            chunk_xyz = xyz[:, start:end]
            chunk_arc = arc_params[:, start:end]
            hybrid_dist = self._compute_hybrid_distance(chunk_xyz, xyz, chunk_arc, arc_params, alpha_clamped)
            _, idx = hybrid_dist.topk(self.k, dim=-1, largest=False)
            knn_idx[:, start:end] = idx
        return knn_idx


class EncoderStage(nn.Module):
    def __init__(self, npoint, k, d_in, d_out, n_blocks, n_heads=4, alpha_init=0.7, dropout=0.1,
                 chunk_size=512):
        super().__init__()
        self.npoint = npoint
        self.knn = HybridKNN(k, alpha_init, chunk_size=chunk_size)
        self.input_norm = nn.LayerNorm(d_in) if d_in == d_out else nn.Identity()
        self.input_proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.blocks = nn.ModuleList([
            ManifoldVecAttnBlock(d_out, n_heads, dropout, drop_path=0.05 * (i + 1) / n_blocks)
            for i in range(n_blocks)
        ])

    def forward(self, x, pos, arc_params):
        fps_idx = furthest_point_sample(pos, self.npoint)
        x_new = index_points(x, fps_idx)
        pos_new = index_points(pos, fps_idx)
        arc_new = index_points(arc_params, fps_idx)
        x_new = self.input_norm(x_new)
        x_new = self.input_proj(x_new)
        knn_idx = self.knn(pos_new, arc_new)
        for block in self.blocks:
            x_new = block(x_new, pos_new, arc_new, knn_idx)
        return x_new, pos_new, arc_new


class ToothQueryCrossAttention(nn.Module):
    def __init__(self, num_queries=53, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model
        self.tooth_embeddings = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, pos=None):
        B = x.shape[0]
        queries = self.tooth_embeddings.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(queries, x, x)
        queries = self.norm(queries + attn_out)
        queries = queries + self.ffn(self.norm2(queries))
        return queries


class ToothContextDecoder(nn.Module):
    def __init__(self, d_concat, d_tooth=512):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(d_concat, 128),
            nn.GELU(approximate='tanh')
        )
        self.score_proj = nn.Linear(128, 53)

    def forward(self, tooth_context, x):
        proj = self.input_proj(x)
        attn_logits = self.score_proj(proj)
        attn_weights = F.softmax(attn_logits, dim=-1)
        return torch.bmm(attn_weights, tooth_context)


class DecoderStage(nn.Module):
    def __init__(self, d_in, d_out, d_skip, d_tooth):
        super().__init__()
        d_concat = d_in + d_skip
        self.tooth_attn = ToothContextDecoder(d_concat, d_tooth)
        self.norm = nn.LayerNorm(d_concat + d_tooth)
        self.mlp = nn.Sequential(
            nn.Linear(d_concat + d_tooth, d_out),
            nn.GELU(approximate='tanh'),
            nn.Dropout(0.1),
            nn.Linear(d_out, d_out),
            nn.GELU(approximate='tanh')
        )

    def forward(self, x, pos, pos_ref, skip_feats, tooth_context):
        interp = three_nn_interpolate(pos, pos_ref, x)
        if skip_feats is not None:
            interp = torch.cat([interp, skip_feats], dim=-1)
        tooth_feat = self.tooth_attn(tooth_context, interp)
        cat_feat = torch.cat([interp, tooth_feat], dim=-1)
        cat_feat = self.norm(cat_feat)
        return self.mlp(cat_feat)


class AuxSegHead(nn.Module):
    def __init__(self, d_in, num_classes=53):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.head = nn.Sequential(
            nn.Linear(d_in, d_in // 2),
            nn.GELU(approximate='tanh'),
            nn.Linear(d_in // 2, num_classes)
        )

    def forward(self, x):
        return self.head(self.norm(x))


class MultiScaleSegHead(nn.Module):
    def __init__(self, d_list, num_classes=53):
        super().__init__()
        d_total = sum(d_list)
        self.head = nn.Sequential(
            nn.Conv1d(d_total, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, feats_list):
        return self.head(torch.cat(feats_list, dim=1))


class DualStreamEncoder(nn.Module):
    def __init__(self, d_out=128):
        super().__init__()
        self.geom_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(approximate='tanh'),
            nn.Linear(64, 64),
            nn.GELU(approximate='tanh')
        )
        self.app_encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.GELU(approximate='tanh'),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.GELU(approximate='tanh'),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.GELU(approximate='tanh')
        )
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64, d_out),
            nn.GELU(approximate='tanh')
        )

    def forward(self, pos, x, arc_params):
        geom_input = torch.cat([pos, arc_params], dim=-1)
        geom_feat = self.geom_encoder(geom_input)
        app_feat = self.app_encoder(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.fusion(torch.cat([geom_feat, app_feat], dim=-1))


class DentalMAN(nn.Module):
    def __init__(self, num_classes=53, d_model=256, n_heads=4, num_points=8192, dropout=0.1,
                 chunk_size=512):
        super().__init__()
        self.num_points = num_points
        self.d_model = d_model

        self.input_encoder = DualStreamEncoder(d_out=128)

        self.encoder1 = EncoderStage(
            npoint=num_points // 4, k=24, d_in=128, d_out=128,
            n_blocks=2, n_heads=n_heads, alpha_init=0.7, dropout=dropout,
            chunk_size=chunk_size
        )
        self.encoder2 = EncoderStage(
            npoint=num_points // 16, k=20, d_in=128, d_out=256,
            n_blocks=2, n_heads=n_heads, alpha_init=0.5, dropout=dropout,
            chunk_size=chunk_size
        )
        self.encoder3 = EncoderStage(
            npoint=num_points // 64, k=16, d_in=256, d_out=512,
            n_blocks=3, n_heads=n_heads, alpha_init=0.3, dropout=dropout,
            chunk_size=chunk_size
        )

        self.bottleneck = ToothQueryCrossAttention(
            num_queries=num_classes, d_model=512, n_heads=8, dropout=dropout
        )

        self.decoder1 = DecoderStage(d_in=512, d_out=256, d_skip=256, d_tooth=512)
        self.decoder2 = DecoderStage(d_in=256, d_out=128, d_skip=128, d_tooth=512)
        self.decoder3 = DecoderStage(d_in=128, d_out=64, d_skip=128, d_tooth=512)

        self.aux_head2 = AuxSegHead(256, num_classes)
        self.aux_head3 = AuxSegHead(128, num_classes)

        self.seg_head = MultiScaleSegHead([256, 128, 64], num_classes)

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.Conv1d)):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

    def forward(self, pos, x, arc_params=None):
        B, N, _ = pos.shape

        if arc_params is None:
            arch_arange = torch.linspace(0, 1, N, device=pos.device).unsqueeze(0).unsqueeze(-1)
            arc_params = arch_arange.expand(B, -1, -1)

        x = self.input_encoder(pos, x, arc_params)

        x1, pos1, arc1 = self.encoder1(x, pos, arc_params)
        x2, pos2, arc2 = self.encoder2(x1, pos1, arc1)
        x3, pos3, arc3 = self.encoder3(x2, pos2, arc2)

        tooth_context = self.bottleneck(x3)

        d1 = self.decoder1(x3, pos2, pos3, x2, tooth_context)
        d2 = self.decoder2(x2, pos1, pos2, x1, tooth_context)
        d3 = self.decoder3(x1, pos, pos1, x, tooth_context)

        aux_logits2_raw = self.aux_head2(d1)
        aux_logits3_raw = self.aux_head3(d2)
        aux_logits2 = three_nn_interpolate(pos, pos2, aux_logits2_raw)
        aux_logits3 = three_nn_interpolate(pos, pos1, aux_logits3_raw)

        d1_up = three_nn_interpolate(pos, pos2, d1)
        d2_up = three_nn_interpolate(pos, pos1, d2)
        d1_t = d1_up.permute(0, 2, 1)
        d2_t = d2_up.permute(0, 2, 1)
        d3_t = d3.permute(0, 2, 1)

        logits = self.seg_head([d1_t, d2_t, d3_t])

        if self.training:
            return logits.permute(0, 2, 1), aux_logits2, aux_logits3
        return logits.permute(0, 2, 1)
