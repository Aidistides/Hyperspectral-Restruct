import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt

# Reuse the same normalisation logic as dataset.py (band-wise z-score)
def hyperspectral_normalize(cube: np.ndarray) -> np.ndarray:
    """Band-wise z-score exactly as in your HyperspectralSoilDataset."""
    mean = np.nanmean(cube, axis=(1, 2), keepdims=True)
    std = np.nanstd(cube, axis=(1, 2), keepdims=True) + 1e-8
    return (cube - mean) / std

def create_variability_map(hsi_cube: np.ndarray) -> np.ndarray:
    """Mimics SoilOptix "countrate" map: first PCA component explains maximum variance."""
    flat = hsi_cube.reshape(hsi_cube.shape[0], -1).T
    pca = PCA(n_components=1)
    var_map = pca.fit_transform(flat).reshape(hsi_cube.shape[1], hsi_cube.shape[2])
    return var_map

def save_variability_tiff(var_map: np.ndarray, transform, crs, path: str):
    with rasterio.open(path, 'w', driver='GTiff', height=var_map.shape[0],
                       width=var_map.shape[1], count=1, dtype=var_map.dtype,
                       crs=crs, transform=transform) as dst:
        dst.write(var_map, 1)
