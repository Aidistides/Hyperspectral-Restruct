import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import zoom, gaussian_filter, map_coordinates
import albumentations as A  # pip install albumentations

class HyperspectralTransform:
    def __init__(self, target_size=(64, 64)):
        self.spatial_aug = A.Compose([
            A.RandomResizedCrop(target_size[0], target_size[1], scale=(0.8, 1.0), p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),  # README elastic deformation
        ])

    def __call__(self, cube):  # cube: (Bands, H, W)
        # Spectral masking (1-20 contiguous bands)
        if np.random.rand() < 0.6:
            start = np.random.randint(0, cube.shape[0] - 20)
            width = np.random.randint(1, 21)
            cube[start:start+width] = 0

        # Spatial masking
        if np.random.rand() < 0.4:
            mask_h, mask_w = np.random.randint(8, 25, 2)
            x1 = np.random.randint(0, cube.shape[1] - mask_h)
            y1 = np.random.randint(0, cube.shape[2] - mask_w)
            cube[:, x1:x1+mask_h, y1:y1+mask_w] = 0

        # Elastic + spatial augs (apply to each band)
        cube = cube.transpose(1, 2, 0)  # (H, W, Bands) for albumentations
        augmented = self.spatial_aug(image=cube)["image"]
        return augmented.transpose(2, 0, 1)  # back to (Bands, H, W)

class HyperspectralSoilDataset(Dataset):
    def __init__(self, data_paths: list, labels: list, num_bands=200, target_size=(64,64), train=True):
        self.data_paths = data_paths
        self.labels = labels  # list of (health_class, contaminant_vector)
        self.transform = HyperspectralTransform(target_size) if train else None
        self.num_bands = num_bands
        self.target_size = target_size

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # Load .npy cube (Bands, H, W) or adapt for .tif / .mat
        cube = np.load(self.data_paths[idx]).astype(np.float32)  # shape (B, H, W)

        # Dimension validation + bilinear resize (README)
        if cube.shape[0] != self.num_bands:
            cube = zoom(cube, (self.num_bands / cube.shape[0], 1, 1), order=1)
        if cube.shape[1:] != self.target_size:
            cube = zoom(cube, (1, self.target_size[0]/cube.shape[1], self.target_size[1]/cube.shape[2]), order=1)

        # Band-wise z-score normalization
        mean = cube.mean(axis=(1,2), keepdims=True)
        std = cube.std(axis=(1,2), keepdims=True) + 1e-8
        cube = (cube - mean) / std

        if self.transform:
            cube = self.transform(cube)

        cube = torch.from_numpy(cube).unsqueeze(0)  # (1, Bands, H, W)
        health_label, contam_label = self.labels[idx]
        return cube, torch.tensor(health_label, dtype=torch.long), torch.tensor(contam_label, dtype=torch.float)
