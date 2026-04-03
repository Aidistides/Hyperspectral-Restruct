import torch
import torch.nn as nn

class SpectralCNN(nn.Module):
    def __init__(self, input_length):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch, bands)
        x = x.unsqueeze(1)  # → (batch, 1, bands)

        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)

        return self.fc(x)
