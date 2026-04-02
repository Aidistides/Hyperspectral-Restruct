import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual3DBlock(nn.Module):
    """
    Standard residual block for 3D convolutions.
    Shortcut projection applied whenever in_channels != out_channels or stride != (1,1,1).
    """

    def __init__(self, in_channels: int, out_channels: int, stride=(1, 1, 1)):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(3, 3, 3), stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        needs_proj = (in_channels != out_channels) or (stride != (1, 1, 1))
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), stride=stride, bias=False),
            nn.BatchNorm3d(out_channels)
        ) if needs_proj else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class SoilHSI3DCNN(nn.Module):
    """
    3D CNN for hyperspectral soil analysis with two output heads:
        - health:  multi-class classification (soil health stages)
        - contam:  multi-label sigmoid output (contaminant presence)

    Key design decisions vs. original:
    ──────────────────────────────────
    1. Spectral downsampling  — layer1 uses stride (2,2,2) so the spectral
       dimension is halved at the first residual stack (200 → 100), preventing
       200 band-planes from passing through all layers unchanged and then being
       brute-force collapsed by AdaptiveAvgPool3d.  Subsequent layers keep the
       spectral stride at 1 so spatial detail can still be reduced independently.

    2. Correct dropout placement — nn.Dropout3d is only appropriate during the
       convolutional feature-learning phase (it zeros whole channel planes).
       After flattening into a 2D vector we switch to standard nn.Dropout so
       individual units, not spatial planes, are regularised.

    3. FC bottleneck — the original 4 096-input head had no hidden layer,
       risking overfitting on small soil datasets.  A two-layer bottleneck
       (4 096 → 512 → num_classes) with ReLU + Dropout in the middle is added
       for both task heads.

    Forward input shape: (B, 1, num_bands, H, W)
    """

    def __init__(
        self,
        num_bands: int = 200,
        num_classes: int = 5,
        num_contaminants: int = 4,
        bottleneck_dim: int = 512,
        dropout_p: float = 0.5,
    ):
        super().__init__()

        # ── Stem ──────────────────────────────────────────────────────────────
        self.initial = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        # ── Residual stacks ───────────────────────────────────────────────────
        # FIX 1 — spectral stride: layer1 uses (2,2,2) to downsample the band
        # axis from 200 → 100.  Subsequent layers keep spectral stride=1 so we
        # can steer spatial resolution reduction independently.
        self.layer1 = self._make_layer(32,  64,  blocks=2, stride=(2, 2, 2))  # bands: /2
        self.layer2 = self._make_layer(64,  128, blocks=2, stride=(1, 2, 2))  # spatial only
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=(1, 2, 2))  # spatial only

        # ── Spatial pooling ───────────────────────────────────────────────────
        # Pool spectral dim to 1, keep a small spatial footprint before the head.
        self.avgpool = nn.AdaptiveAvgPool3d((1, 4, 4))

        # FIX 2 — spatial Dropout3d in the conv stack, NOT after flattening.
        # This zeroes full channel planes while feature maps are still spatial.
        self.spatial_dropout = nn.Dropout3d(p=dropout_p)

        # After avgpool + flatten: (B, 256*1*4*4) = (B, 4096)
        flat_dim = 256 * 1 * 4 * 4  # 4 096

        # FIX 3 — bottleneck head shared between the two task-specific layers.
        # Linear(4096 → 512) compresses the representation before the task
        # outputs, significantly reducing the parameter count and overfitting risk.
        self.bottleneck = nn.Sequential(
            nn.Linear(flat_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            # FIX 2 (continued) — standard Dropout after flattening, not Dropout3d.
            nn.Dropout(p=dropout_p),
        )

        # Task heads
        self.fc_health = nn.Linear(bottleneck_dim, num_classes)
        self.fc_contam = nn.Linear(bottleneck_dim, num_contaminants)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, blocks: int, stride) -> nn.Sequential:
        layers = [Residual3DBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(Residual3DBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 1, num_bands, H, W)  — single-channel 3-D hyperspectral cube

        Returns:
            health: (B, num_classes)          raw logits for CE loss
            contam: (B, num_contaminants)     sigmoid probabilities for BCE loss
        """
        x = self.initial(x)          # (B, 32,  bands,   H,   W)

        x = self.layer1(x)           # (B, 64,  bands/2, H/2, W/2)  ← spectral ↓
        x = self.spatial_dropout(x)  # channel-plane dropout while still spatial
        x = self.layer2(x)           # (B, 128, bands/2, H/4, W/4)
        x = self.layer3(x)           # (B, 256, bands/2, H/8, W/8)

        x = self.avgpool(x)          # (B, 256, 1, 4, 4)
        x = torch.flatten(x, 1)      # (B, 4096)

        shared = self.bottleneck(x)  # (B, 512)  ← bottleneck + standard Dropout

        health = self.fc_health(shared)              # (B, num_classes)   — raw logits
        contam = torch.sigmoid(self.fc_contam(shared))  # (B, num_contaminants) — probs

        return health, contam


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = SoilHSI3DCNN(num_bands=200, num_classes=5, num_contaminants=4)
    model.eval()

    dummy = torch.randn(2, 1, 200, 64, 64)  # batch=2, 1 channel, 200 bands, 64×64
    with torch.no_grad():
        health_logits, contam_probs = model(dummy)

    print("health logits shape :", health_logits.shape)   # (2, 5)
    print("contam probs  shape :", contam_probs.shape)    # (2, 4)
    print("contam range  [0,1] :", contam_probs.min().item(), "–", contam_probs.max().item())

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
