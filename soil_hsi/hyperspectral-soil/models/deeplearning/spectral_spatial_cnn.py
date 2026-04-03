import torch
import torch.nn as nn

class SpectralSpatialCNN(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3,3,7), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )

        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        # x shape: (batch, height, width, bands)
        x = x.unsqueeze(1)  # → (batch, 1, H, W, B)

        x = self.conv3d(x)
        x = x.view(x.size(0), -1)

        return self.fc(x)
