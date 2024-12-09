import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    # (Convolution + ReLU) * 2
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class AttentionBlock(nn.Module):
    # Attention Block for UNet: Gating Signal concatenation->convolution
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1),
            nn.BatchNorm1d(F_int),
            nn.ReLU(inplace=True)
        )

        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1),
            nn.BatchNorm1d(F_int),
            nn.ReLU(inplace=True)
        )

        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)

        # Up-sample x1 from 128 to 256 before adding to g1
        x1 = self.W_x(x)
        # Ensure that the target size is specified
        target_size = g1.size()[2]  # Grab the size from g1 for consistency
        x1 = F.interpolate(x1, size=target_size, mode='linear', align_corners=True)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        target_size = psi.size()[2]
        x = F.interpolate(x, size=target_size, mode='linear', align_corners=True)
        return x * psi


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Define architecture here
        # Encoder path
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DoubleConv(64, 128)
        ...  # Add down-sampling layers

        # Decoder path: Up-sampling plus double conv
        ...  # Add up-sampling layers
        self.up1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.dec1 = DoubleConv(128, 64)

        # Out Conv (linear layer to map the results to the output)
        # Out Conv (linear layer to map the results to the output)
        self.outc = nn.Conv1d(64, n_classes, kernel_size=1025)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        ...  # Pass through down-sampling layers

        # Decoder path
        ...  # Pass through up-sampling layers
        x = self.up1(x2)
        y = self.att1(g=x, x=x1)
        x = torch.cat([x, y], dim=1)
        x = self.dec1(x)

        # Output mapping
        logits = self.outc(x)
        return logits


#
# model = UNet(n_channels=1, n_classes=1)  # single channel input and output
#
# # Dummy input tensor with the shape (batch_size, channels, width)
# x = torch.randn(4, 1, 1024)
# # Forward pass through the network
# out = model(x)
# print(out.shape)