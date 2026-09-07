import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetSegmentation(nn.Module):
    def __init__(self, num_classes, in_channels=3, dropout_rate=0.5):
        super(PointNetSegmentation, self).__init__()
        # Feature Extraction
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.conv5 = nn.Conv1d(512, 1024, 1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)

        # Segmentation head
        # Input size: 1024 (global) + 512 + 256 (local) = 1792
        self.fc1 = nn.Linear(1792, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc_bn2 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        # Feature extraction
        x1 = F.relu(self.bn1(self.conv1(x)))  # [B, 64, N]
        x2 = F.relu(self.bn2(self.conv2(x1)))  # [B, 128, N]
        x3 = F.relu(self.bn3(self.conv3(x2)))  # [B, 256, N]
        x4 = F.relu(self.bn4(self.conv4(x3)))  # [B, 512, N]
        x5 = F.relu(self.bn5(self.conv5(x4)))  # [B, 1024, N]

        # Global feature
        global_feat = torch.max(x5, 2, keepdim=True)[0]  # [B, 1024, 1]
        global_feat_expanded = global_feat.repeat(1, 1, num_points)  # [B, 1024, N]

        # Concatenate global and local features
        # Using skip connections from earlier layers
        concat_feat = torch.cat([global_feat_expanded, x4, x3], 1)  # [B, 1792, N]

        # Segmentation head
        net = concat_feat.transpose(2, 1)  # [B, N, 1792]
        net = F.relu(self.fc_bn1(self.fc1(net).transpose(1, 2)).transpose(1, 2))
        net = self.dropout(net)

        net = F.relu(self.fc_bn2(self.fc2(net).transpose(1, 2)).transpose(1, 2))
        net = self.dropout(net)

        net = self.fc3(net)  # [B, N, num_classes]

        return net.transpose(2, 1)  # [B, num_classes, N]

    def get_loss(self, pred, target, smoothing=True):
        """Calculate cross entropy loss with label smoothing"""
        target = target.contiguous().view(-1)

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            pred = pred.transpose(1, 2).contiguous().view(-1, n_class)  # [B*N, num_classes]
            one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            pred = pred.transpose(1, 2).contiguous().view(-1, pred.size(1))
            loss = F.cross_entropy(pred, target)

        return loss