# soil_hsi/datasets/hyperspectral_soil.py

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple

# Try to import your existing preprocessing if available
try:
    from ..preprocessing import PreprocessingPipeline  # adjust import based on actual location
except ImportError:
    PreprocessingPipeline = None


class HyperspectralSoilDataset(Dataset):
    """
    Custom Dataset for hyperspectral soil / microplastics data.
    Supports:
      - .npy files (1D spectra or 2D/3D patches)
      - CSV label file (recommended)
      - Optional preprocessing pipeline from your repo
    """

    def __init__(
        self,
        data_dir: str,                                   # e.g. "data/train/patches" or "data/spectra"
        label_file: Optional[str] = None,                # path to labels.csv
        transform: Optional = None,                      # PreprocessingPipeline or torchvision transforms
        target_transform=None,
        wavelength_crop: Optional[Tuple[int, int]] = None,
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.wavelength_crop = wavelength_crop

        if label_file and os.path.exists(label_file):
            # Preferred way: CSV with filename → label mapping
            df = pd.read_csv(label_file)
            # Expected columns: at minimum 'filename' and 'label'
            # For regression (soil pH, nutrients, etc.) you can have multiple target columns
            self.file_names = df['filename'].tolist()
            self.labels = df['label'].values.astype(np.float32)  # change dtype if classification (int)
            self.data_paths = [os.path.join(data_dir, fname) for fname in self.file_names]
        else:
            # Fallback: glob all .npy files (useful for quick testing)
            self.data_paths = sorted(glob.glob(os.path.join(data_dir, "**/*.npy"), recursive=True))
            self.labels = np.zeros(len(self.data_paths), dtype=np.float32)  # dummy

        if len(self.data_paths) == 0:
            raise RuntimeError(f"No .npy files found in {data_dir}")

        print(f"Loaded {len(self)} samples from {data_dir}")

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx: int):
        # Load the hyperspectral data
        data = np.load(self.data_paths[idx]).astype(np.float32)

        # Optional: crop wavelength bands (if you know the indices)
        if self.wavelength_crop is not None:
            start, end = self.wavelength_crop
            data = data[..., start:end]   # works for both 1D and multi-dim

        # Apply preprocessing (SNV, smoothing, continuum removal, etc.)
        if self.transform is not None:
            if PreprocessingPipeline is not None and isinstance(self.transform, PreprocessingPipeline):
                data = self.transform(data)  # adjust call if your pipeline needs extra args
            else:
                data = self.transform(data)

        # Convert to tensor
        data = torch.from_numpy(data)

        # Common shape fixes for models:
        # - 1D spectrum → (1, bands)   for 1D CNN / Transformer
        # - Patch       → (bands, H, W) channels-first (standard in PyTorch)
        if data.ndim == 1:
            data = data.unsqueeze(0)                    # (bands,) → (1, bands)
        elif data.ndim == 3 and data.shape[0] > data.shape[-1]:  # if (H, W, bands) → (bands, H, W)
            data = data.permute(2, 0, 1)

        label = self.labels[idx]
        if self.target_transform:
            label = self.target_transform(label)

        return data, torch.tensor(label, dtype=torch.float32)  # use torch.long for classification


# Optional: helper to create DataLoaders
def get_dataloaders(
    train_dir: str,
    train_label_file: Optional[str] = None,
    val_dir: Optional[str] = None,
    val_label_file: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    transform=None,
):
    train_ds = HyperspectralSoilDataset(train_dir, train_label_file, transform=transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_dir:
        val_ds = HyperspectralSoilDataset(val_dir, val_label_file, transform=transform)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader
