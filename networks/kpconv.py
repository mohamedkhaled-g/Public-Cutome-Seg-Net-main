import torch
import torch.nn as nn
import torch.nn.functional as F

class KPConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, debug=False):
        super(KPConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.debug = debug

    def forward(self, x):
        if self.debug:
            print(f"KPConvBlock input shape: {x.shape}")

        # Transpose the input tensor to [batch_size, channels, sequence_length]
        x = x.permute(0, 2, 1)  # Transpose from [batch_size, sequence_length, channels] to [batch_size, channels, sequence_length]
        if self.debug:
            print(f"After permute: {x.shape}")

        x = self.conv(x)
        if self.debug:
            print(f"After conv: {x.shape}")

        x = self.bn(x)
        if self.debug:
            print(f"After bn: {x.shape}")

        x = self.relu(x)
        if self.debug:
            print(f"After relu: {x.shape}")

        x = x.permute(0, 2, 1)  # Transpose back to [batch_size, sequence_length, channels] for the next layer
        if self.debug:
            print(f"After permute back: {x.shape}")

        return x

class KPConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, debug=False, dropout_rate=0.0):
        super(KPConvNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.debug = debug
        self.dropout_rate = dropout_rate

        # Encoder
        self.encoder1 = KPConvBlock(in_channels, 64, debug=debug)
        self.encoder2 = KPConvBlock(64, 128, debug=debug)
        self.encoder3 = KPConvBlock(128, 256, debug=debug)
        self.encoder4 = KPConvBlock(256, 512, debug=debug)

        # Decoder
        self.decoder1 = KPConvBlock(512, 256, debug=debug)
        self.decoder2 = KPConvBlock(256, 128, debug=debug)
        self.decoder3 = KPConvBlock(128, 64, debug=debug)
        self.decoder4 = KPConvBlock(64, out_channels, debug=debug)

        # Final classification layer
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, data):
        if self.debug:
            print(f"Model input data.x shape: {data.x.shape}")

        # Extract features (x) from the Data object
        x = data.x  # Shape: [batch_size * num_points, in_channels]

        # Reshape x to [batch_size, num_points, in_channels]
        batch_size = data.batch.max().item() + 1
        num_points = data.x.size(0) // batch_size
        x = x.reshape(batch_size, num_points, -1)  # Use reshape instead of view
        if self.debug:
            print(f"After reshape: {x.shape}")

        # Encoder
        x1 = self.encoder1(x)  # Shape: [batch_size, num_points, channels]
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        # Decoder
        x = self.decoder1(x4)
        x = self.decoder2(x + x3)  # Skip connection
        x = self.decoder3(x + x2)  # Skip connection
        x = self.decoder4(x + x1)  # Skip connection

        # Reshape output to [batch_size * num_points, out_channels]
        x = x.reshape(batch_size * num_points, -1)  # Use reshape instead of view
        if self.debug:
            print(f"After final reshape: {x.shape}")

        # Dropout before classification (mirrors the PointNet recipe)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Final classification
        x = self.fc(x)  # Shape: [batch_size * num_points, num_classes]
        if self.debug:
            print(f"After fc: {x.shape}")

        return x