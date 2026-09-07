import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channels, mlp):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        last_channels = in_channels + 3  # +3 for xyz coordinates
        for out_channels in mlp:
            self.convs.append(nn.Conv2d(last_channels, out_channels, 1))
            self.bns.append(nn.BatchNorm2d(out_channels))
            last_channels = out_channels

    def forward(self, xyz, points):
        """
        xyz: [B, 3, N]
        points: [B, C, N] or None
        Returns: new_xyz [B, 3, npoint], new_points [B, C', npoint]
        """
        B, C, N = xyz.shape
        # print(f"SA: xyz shape: {xyz.shape}, points shape: {points.shape if points is not None else None}")
        assert C == 3, f"Expected xyz to have 3 channels, got {C}"

        if self.npoint is not None:
            # Farthest Point Sampling
            idx = farthest_point_sample(xyz, self.npoint)  # [B, npoint]
            # print(f"SA: idx from FPS shape: {idx.shape}")
            new_xyz = index_points(xyz, idx)  # [B, 3, npoint]
            # print(f"SA: new_xyz shape: {new_xyz.shape}")
            assert new_xyz.shape == (B, 3, self.npoint), f"new_xyz shape: {new_xyz.shape}"

            # Ball Query
            group_idx = ball_query(new_xyz, xyz, self.radius, self.nsample)  # [B, npoint, nsample]
            # print(f"SA: group_idx shape: {group_idx.shape}")
            grouped_xyz = index_points(xyz, group_idx)  # [B, 3, npoint, nsample]
            # print(f"SA: grouped_xyz shape: {grouped_xyz.shape}")
            grouped_xyz -= new_xyz.unsqueeze(-1)  # [B, 3, npoint, nsample]
            # print(f"SA: grouped_xyz after subtraction shape: {grouped_xyz.shape}")

            if points is not None:
                grouped_points = index_points(points, group_idx)  # [B, C, npoint, nsample]
                # print(f"SA: grouped_points shape: {grouped_points.shape}")
                new_points = torch.cat([grouped_xyz, grouped_points], dim=1)  # [B, C+3, npoint, nsample]
                # print(f"SA: new_points after concat shape: {new_points.shape}")
            else:
                new_points = grouped_xyz
        else:
            # Global Set Abstraction: Use all points, no sampling
            new_xyz = torch.zeros(B, 3, 1, device=xyz.device)  # Dummy centroid [B, 3, 1]
            grouped_xyz = xyz.unsqueeze(-1)  # [B, 3, N, 1]
            # print(f"SA: grouped_xyz shape (global): {grouped_xyz.shape}")
            if points is not None:
                grouped_points = points.unsqueeze(-1)  # [B, C, N, 1]
                # print(f"SA: grouped_points shape (global): {grouped_points.shape}")
                new_points = torch.cat([grouped_xyz, grouped_points], dim=1)  # [B, C+3, N, 1]
                # print(f"SA: new_points after concat (global): {new_points.shape}")
            else:
                new_points = grouped_xyz

        # PointNet-like processing
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            new_points = F.relu(bn(conv(new_points)))
            # print(f"SA: new_points after conv {i+1} shape: {new_points.shape}")

        new_points = torch.max(new_points, -1)[0]  # [B, C', npoint] or [B, C', N]
        # print(f"SA: new_points after max shape: {new_points.shape}")
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channels, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        last_channels = in_channels
        for out_channels in mlp:
            self.convs.append(nn.Conv1d(last_channels, out_channels, 1))
            self.bns.append(nn.BatchNorm1d(out_channels))
            last_channels = out_channels

    def forward(self, xyz1, xyz2, points1, points2):
        """
        xyz1: [B, 3, N1] (denser)
        xyz2: [B, 3, N2] (sparser)
        points1: [B, C1, N1] or None
        points2: [B, C2, N2]
        Returns: [B, C', N1]
        """
        # print(f"FP: xyz1 shape: {xyz1.shape}, xyz2 shape: {xyz2.shape}")
        # print(f"FP: points1 shape: {points1.shape if points1 is not None else None}, points2 shape: {points2.shape}")
        dists, idx = three_nn(xyz1, xyz2)  # [B, N1, k], [B, N1, k]
        # print(f"FP: dists shape: {dists.shape}, idx shape: {idx.shape}")
        dists = torch.clamp(dists, min=1e-10)
        weights = 1.0 / dists
        weights = weights / weights.sum(dim=-1, keepdim=True)  # Normalize weights
        # print(f"FP: weights shape: {weights.shape}")

        interpolated_points = three_interpolate(points2, idx, weights)  # [B, C2, N1]
        # print(f"FP: interpolated_points shape: {interpolated_points.shape}")

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=1)
            # print(f"FP: new_points after concat shape: {new_points.shape}")
        else:
            new_points = interpolated_points

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            new_points = F.relu(bn(conv(new_points)))
            # print(f"FP: new_points after conv {i+1} shape: {new_points.shape}")

        return new_points


class PointNetPlusPlus(nn.Module):
    def __init__(self, num_classes, in_channels=3, dropout_rate=0.5):
        super(PointNetPlusPlus, self).__init__()
        self.sa1 = PointNetSetAbstraction(512, 0.1, 32, in_channels, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(128, 0.2, 32, 128, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(None, None, None, 256, [256, 512, 1024])  # Global SA

        self.fp3 = PointNetFeaturePropagation(1024 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(256 + 128, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128 + in_channels, [128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        """
        xyz: [B, 6, N] (xyz + rgb, as produced by the paper preprocessing)
        Returns: [B, num_classes, N]
        """
        # print(f"PN++: Input xyz shape: {xyz.shape}")
        l0_xyz = xyz[:, :3]
        l0_points = xyz[:, 3:]  # rgb used as initial point features

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # print(f"PN++: l1_xyz shape: {l1_xyz.shape}, l1_points shape: {l1_points.shape}")
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)
        # print(f"PN++: l0_points after fp1 shape: {l0_points.shape}")

        net = F.relu(self.bn1(self.conv1(l0_points)))
        # print(f"PN++: net after conv1 shape: {net.shape}")
        net = self.dropout(net)
        net = self.conv2(net)
        # print(f"PN++: Output shape: {net.shape}")

        return net

    def get_loss(self, pred, target, smoothing=True):
        target = target.contiguous().view(-1)
        if smoothing:
            eps = 0.2
            n_class = pred.size(1)
            pred = pred.transpose(1, 2).contiguous().view(-1, n_class)
            one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)
            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            pred = pred.transpose(1, 2).contiguous().view(-1, pred.size(1))
            loss = F.cross_entropy(pred, target)
        return loss


# Helper functions
def farthest_point_sample(xyz, npoint):
    """
    xyz: [B, 3, N]
    Returns: [B, npoint]
    """
    B, _, N = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), device=device)
    batch_indices = torch.arange(B, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, :, farthest].unsqueeze(-1)  # [B, 3, 1]
        dist = torch.sum((xyz - centroid) ** 2, dim=1)  # [B, N]
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance, dim=1)
    return centroids


def index_points(points, idx):
    """
    points: [B, C, N]
    idx: [B, S] or [B, S, K]
    Returns: [B, C, S] or [B, C, S, K]
    """
    device = points.device
    B, C, N = points.shape
    idx_shape = idx.shape
    # print(f"index_points: points shape: {points.shape}, idx shape: {idx.shape}")
    idx = idx.reshape(B, -1)  # Flatten idx to [B, S] or [B, S*K]
    # print(f"index_points: idx after reshape shape: {idx.shape}")
    batch_indices = torch.arange(B, device=device).view(B, 1).expand(B, idx.size(1))
    # print(f"index_points: batch_indices shape: {batch_indices.shape}")
    new_points = points[batch_indices, :, idx]  # [B, S or S*K, C]
    # print(f"index_points: new_points after indexing shape: {new_points.shape}")
    if len(idx_shape) == 3:  # If idx was [B, S, K]
        new_points = new_points.reshape(B, idx_shape[1], idx_shape[2], C).permute(0, 3, 1, 2)  # [B, C, S, K]
    else:  # If idx was [B, S]
        new_points = new_points.permute(0, 2, 1)  # [B, C, S]
    # print(f"index_points: new_points final shape: {new_points.shape}")
    return new_points


def ball_query(new_xyz, xyz, radius, nsample):
    """
    new_xyz: [B, 3, npoint]
    xyz: [B, 3, N]
    Returns: [B, npoint, nsample]
    """
    B, C, npoint = new_xyz.shape
    _, _, N = xyz.shape
    assert C == 3, f"Expected 3 channels, got {C}"
    # print(f"ball_query: new_xyz shape: {new_xyz.shape}, xyz shape: {xyz.shape}")
    sqrdists = square_distance(new_xyz, xyz)  # [B, npoint, N]
    # print(f"ball_query: sqrdists shape: {sqrdists.shape}")
    device = sqrdists.device

    group_idx = torch.arange(N, device=device).view(1, 1, N).repeat(B, npoint, 1)  # [B, npoint, N]
    # print(f"ball_query: group_idx initial shape: {group_idx.shape}")
    mask = sqrdists > radius ** 2
    group_idx[mask] = N  # Mark out-of-range points
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # Take closest nsample points
    # print(f"ball_query: group_idx after sort shape: {group_idx.shape}")

    mask = group_idx == N
    if mask.any():
        random_idx = torch.randint(0, N, (B, npoint, nsample), device=device)
        group_idx[mask] = random_idx[mask]
    # print(f"ball_query: group_idx final shape: {group_idx.shape}")

    return group_idx


def square_distance(src, dst):
    """
    src: [B, C, N1] (xyz1)
    dst: [B, C, N2] (xyz2)
    Returns: [B, N1, N2]
    """
    B, C, N1 = src.shape
    _, _, N2 = dst.shape
    assert C == 3, f"Expected 3 channels, got {C}"
    # print(f"square_distance: src shape: {src.shape}, dst shape: {dst.shape}")

    src_expanded = src.unsqueeze(3)  # [B, C, N1, 1]
    dst_expanded = dst.unsqueeze(2)  # [B, C, 1, N2]
    # print(f"square_distance: src_expanded shape: {src_expanded.shape}, dst_expanded shape: {dst_expanded.shape}")

    diff = src_expanded - dst_expanded  # [B, C, N1, N2]
    # print(f"square_distance: diff shape: {diff.shape}")
    dist = torch.sum(diff ** 2, dim=1)  # [B, N1, N2]
    # print(f"square_distance: dist shape: {dist.shape}")
    return dist


def three_nn(xyz1, xyz2):
    """
    xyz1: [B, 3, N1]
    xyz2: [B, 3, N2]
    Returns: dists [B, N1, k], idx [B, N1, k], where k = min(3, N2)
    """
    dists = square_distance(xyz1, xyz2)  # [B, N1, N2]
    N2 = xyz2.shape[2]
    k = min(3, N2)  # Use the smaller of 3 or N2
    dists, idx = dists.topk(k, dim=-1, largest=False)  # [B, N1, k], [B, N1, k]
    return dists, idx


def three_interpolate(points, idx, weights):
    """
    points: [B, C, N2]
    idx: [B, N1, k]
    weights: [B, N1, k]
    Returns: [B, C, N1]
    """
    # print(f"three_interpolate: points shape: {points.shape}, idx shape: {idx.shape}, weights shape: {weights.shape}")
    grouped_points = index_points(points, idx)  # [B, C, N1, k]
    # print(f"three_interpolate: grouped_points shape: {grouped_points.shape}")
    result = torch.sum(grouped_points * weights.unsqueeze(1), dim=3)  # [B, C, N1]
    # print(f"three_interpolate: result shape: {result.shape}")
    return result