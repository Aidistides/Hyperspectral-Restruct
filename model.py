import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual3DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=(1,1,1)):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3,3,3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3,3,3), stride=(1,1,1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1,1,1), stride=stride, bias=False),
            nn.BatchNorm3d(out_channels)
        ) if in_channels != out_channels or stride != (1,1,1) else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class SoilHSI3DCNN(nn.Module):
    """Elite 3D CNN exactly as described in the README"""
    def __init__(self, num_bands: int = 200, num_classes: int = 5, num_contaminants: int = 4):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3,3,3), stride=(1,1,1), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(32, 64, blocks=2, stride=(1,2,2))
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=(1,2,2))
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=(1,2,2))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 4, 4))  # keeps some spatial context
        self.dropout = nn.Dropout3d(p=0.5)  # progressive spatial dropout
        self.fc_health = nn.Linear(256 * 4 * 4, num_classes)          # soil health stages
        self.fc_contam = nn.Linear(256 * 4 * 4, num_contaminants)     # multi-label contaminants

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [Residual3DBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(Residual3DBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):  # x: (B, 1, Bands, H, W)
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        health = self.fc_health(x)
        contam = torch.sigmoid(self.fc_contam(x))  # multi-label
        return health, contam
