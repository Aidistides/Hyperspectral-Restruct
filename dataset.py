import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom, map_coordinates, gaussian_filter
import numpy as np
from pathlib import Path
import cv2
from typing import List, Tuple, Optional, Union
import warnings

# Import calibration module
try:
    from calibration import CalibrationPipeline, CalibrationConfig
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    warnings.warn("Calibration module not available. Install calibration dependencies for radiometric correction.")


# ── Augmentation pipeline ─────────────────────────────────────────────────────

class HyperspectralTransform:
    """
    Augmentation pipeline for hyperspectral cubes shaped (Bands, H, W).

    Design decisions vs. original
    ──────────────────────────────
    FIX 1 — masking order:
        Spectral and spatial masking are now applied BEFORE band-wise z-score
        normalization (see HyperspectralSoilDataset.__getitem__).  This class
        therefore receives a raw (un-normalised) cube so that zeroing masked
        regions genuinely means "no reflectance signal", not an arbitrary
        post-normalisation artefact.  The fill value used is 0 (raw DN / raw
        reflectance ~0), which is semantically correct at this stage.

    FIX 2 — no albumentations:
        albumentations was designed for uint8 RGB images.  With 200-band
        float32 cubes several internals break silently (dtype coercion,
        channel-count assumptions, ElasticTransform's cv2 backend).  All
        spatial augmentations are re-implemented here using only numpy /
        scipy so behaviour is explicit and band-count agnostic.

        Implemented transforms:
            • Random horizontal / vertical flip
            • Random crop + resize  (replaces RandomResizedCrop)
            • Elastic deformation   (replaces A.ElasticTransform)
              — uses scipy.ndimage.map_coordinates directly, exactly as
                suggested in the original code review.
    """

    def __init__(
        self,
        target_size: tuple = (64, 64),
        # spectral masking
        spec_mask_prob: float = 0.6,
        spec_mask_max_width: int = 20,
        # spatial masking
        spat_mask_prob: float = 0.4,
        spat_mask_min: int = 8,
        spat_mask_max: int = 25,
        # spatial augmentation
        flip_prob: float = 0.5,
        crop_prob: float = 0.8,
        crop_scale: tuple = (0.8, 1.0),
        elastic_prob: float = 0.3,
        elastic_alpha: float = 1.0,
        elastic_sigma: float = 50.0,
    ):
        self.target_size = target_size
        self.spec_mask_prob = spec_mask_prob
        self.spec_mask_max_width = spec_mask_max_width
        self.spat_mask_prob = spat_mask_prob
        self.spat_mask_min = spat_mask_min
        self.spat_mask_max = spat_mask_max
        self.flip_prob = flip_prob
        self.crop_prob = crop_prob
        self.crop_scale = crop_scale
        self.elastic_prob = elastic_prob
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma

    # ── masking (called on RAW cube, before normalisation) ────────────────────

    def spectral_mask(self, cube: np.ndarray) -> np.ndarray:
        """Zero out 1–spec_mask_max_width contiguous bands."""
        if np.random.rand() < self.spec_mask_prob:
            n_bands = cube.shape[0]
            width = np.random.randint(1, self.spec_mask_max_width + 1)
            start = np.random.randint(0, max(1, n_bands - width))
            cube[start: start + width] = 0.0
        return cube

    def spatial_mask(self, cube: np.ndarray) -> np.ndarray:
        """Zero out a random rectangular patch across all bands."""
        if np.random.rand() < self.spat_mask_prob:
            H, W = cube.shape[1], cube.shape[2]
            mask_h = np.random.randint(self.spat_mask_min, self.spat_mask_max + 1)
            mask_w = np.random.randint(self.spat_mask_min, self.spat_mask_max + 1)
            x1 = np.random.randint(0, max(1, H - mask_h))
            y1 = np.random.randint(0, max(1, W - mask_w))
            cube[:, x1: x1 + mask_h, y1: y1 + mask_w] = 0.0
        return cube

    # ── spatial augmentations (band-agnostic numpy/scipy) ─────────────────────

    def random_flip(self, cube: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.flip_prob:
            cube = cube[:, ::-1, :].copy()   # vertical
        if np.random.rand() < self.flip_prob:
            cube = cube[:, :, ::-1].copy()   # horizontal
        return cube

    def random_crop_resize(self, cube: np.ndarray) -> np.ndarray:
        """
        Randomly crop a sub-window then zoom back to target_size.
        Equivalent to albumentations RandomResizedCrop but works for any
        number of bands via scipy.ndimage.zoom.
        """
        if np.random.rand() >= self.crop_prob:
            # Just resize to target if no crop this iteration
            H, W = cube.shape[1], cube.shape[2]
            tH, tW = self.target_size
            if (H, W) != (tH, tW):
                cube = zoom(cube, (1, tH / H, tW / W), order=1)
            return cube

        H, W = cube.shape[1], cube.shape[2]
        tH, tW = self.target_size

        scale = np.random.uniform(*self.crop_scale)
        crop_h = max(1, int(H * scale))
        crop_w = max(1, int(W * scale))

        x0 = np.random.randint(0, max(1, H - crop_h + 1))
        y0 = np.random.randint(0, max(1, W - crop_w + 1))

        cropped = cube[:, x0: x0 + crop_h, y0: y0 + crop_w]
        # Resize back to target
        cube = zoom(cropped, (1, tH / crop_h, tW / crop_w), order=1)
        return cube

    def elastic_deform(self, cube: np.ndarray) -> np.ndarray:
        """
        FIX 2 — elastic deformation implemented directly with
        scipy.ndimage.map_coordinates (the import already existed in the
        original code but was never used).

        A single displacement field is computed for the H×W grid and then
        applied identically to every spectral band so spatial coherence
        across bands is preserved.
        """
        if np.random.rand() >= self.elastic_prob:
            return cube

        _, H, W = cube.shape

        # Random displacement fields, smoothed to look locally coherent
        dx = gaussian_filter(
            np.random.randn(H, W) * self.elastic_alpha,
            sigma=self.elastic_sigma
        )
        dy = gaussian_filter(
            np.random.randn(H, W) * self.elastic_alpha,
            sigma=self.elastic_sigma
        )

        # Build the absolute sampling coordinates
        grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        coords_y = np.clip(grid_y + dy, 0, H - 1).ravel()
        coords_x = np.clip(grid_x + dx, 0, W - 1).ravel()

        # Apply the same warp to every band (preserves spectral coherence)
        deformed = np.empty_like(cube)
        for b in range(cube.shape[0]):
            deformed[b] = map_coordinates(
                cube[b], [coords_y, coords_x], order=1, mode="reflect"
            ).reshape(H, W)

        return deformed

    # ── full pipeline (operates on RAW cube) ──────────────────────────────────

    def __call__(self, cube: np.ndarray) -> np.ndarray:
        """
        Apply masking then spatial augmentations to a raw (Bands, H, W) cube.
        Normalization is intentionally NOT performed here — it happens after
        this call in __getitem__ so masked zeros stay semantically meaningful.
        """
        # FIX 1 — mask first, normalise later
        cube = self.spectral_mask(cube)
        cube = self.spatial_mask(cube)

        # Spatial augmentations (band-count agnostic, no albumentations)
        cube = self.random_flip(cube)
        cube = self.random_crop_resize(cube)   # also handles final resize
        cube = self.elastic_deform(cube)

        return cube


# ── Dataset ───────────────────────────────────────────────────────────────────

class HyperspectralSoilDataset(Dataset):
    """
    PyTorch Dataset for hyperspectral soil cubes stored as .npy files.

    Args:
        data_paths      : list of paths to (Bands, H, W) float32 .npy cubes.
        labels          : list of (health_class: int, contam_vector: list[float]).
        num_bands       : target number of spectral bands (resampled if needed).
        target_size     : (H, W) spatial size every cube is resized to.
        train           : if True, applies HyperspectralTransform augmentation.

    Normalisation note (FIX 1 / FIX 3)
    ────────────────────────────────────
    Band-wise z-score normalization is computed per-sample on the POST-MASKED
    cube.  Per-sample normalisation is correct for HSI because illumination
    and sensor gain vary per acquisition; dataset-level statistics would leak
    train-set information into the validation set and are inappropriate here.

    Order of operations in __getitem__:
        1. Load raw cube
        2. Resample bands / spatial size if needed
        3. Apply augmentation (masking + spatial transforms) — RAW values
        4. Band-wise z-score normalisation
        5. Convert to tensor
    """

    def __init__(
        self,
        data_paths: list,
        labels: list,
        num_bands: int = 200,
        target_size: tuple = (64, 64),
        train: bool = True,
        calibrate: bool = False,
        calib_config: Optional[str] = None,
        wavelengths: Optional[np.ndarray] = None,
    ):
        assert len(data_paths) == len(labels), \
            f"data_paths and labels length mismatch: {len(data_paths)} vs {len(labels)}"

        self.data_paths = data_paths
        self.labels = labels
        self.num_bands = num_bands
        self.target_size = target_size
        self.transform = HyperspectralTransform(target_size) if train else None
        
        # Initialize calibration pipeline
        self.calibrate = calibrate and CALIBRATION_AVAILABLE
        self.calibration_pipeline = None
        
        if self.calibrate:
            if calib_config is None:
                # Use default calibration config
                calib_config = "calibration/configs/default.yaml"
            
            if wavelengths is None:
                # Default wavelength range (400-1000nm, 3nm steps for 200 bands)
                wavelengths = np.linspace(400, 1000, num_bands)
            
            try:
                config = CalibrationConfig.from_yaml(calib_config)
                self.calibration_pipeline = CalibrationPipeline(config, wavelengths)
                self.wavelengths = wavelengths
                print(f"Calibration pipeline initialized with {len(wavelengths)} wavelengths")
            except Exception as e:
                print(f"Failed to initialize calibration pipeline: {e}")
                self.calibrate = False

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, idx: int):
        # Step 1: Load raw cube
        cube = np.load(self.data_paths[idx]).astype(np.float32)  # (B, H, W)

        # Step 2: Apply calibration if enabled (before other processing)
        if self.calibrate and self.calibration_pipeline is not None:
            try:
                # Transpose to (H, W, B) for calibration pipeline
                cube_hwb = cube.transpose(1, 2, 0)
                calibrated_cube, _ = self.calibration_pipeline.calibrate_single_cube(cube_hwb)
                # Transpose back to (B, H, W)
                cube = calibrated_cube.transpose(2, 0, 1)
            except Exception as e:
                print(f"Calibration failed for sample {idx}: {e}")
                # Continue with uncalibrated data

        # Step 3: Resample to canonical shape
        # Band dimension
        if cube.shape[0] != self.num_bands:
            cube = zoom(cube, (self.num_bands / cube.shape[0], 1, 1), order=1)
        # Spatial dimensions (only if not training, where crop_resize handles it)
        if not self.transform:
            H, W = cube.shape[1], cube.shape[2]
            tH, tW = self.target_size
            if (H, W) != (tH, tW):
                cube = zoom(cube, (1, tH / H, tW / W), order=1)

        # Step 4: Augmentation on RAW cube (masking + spatial)
        if self.transform is not None:
            cube = self.transform(cube)

        # Step 5: Band-wise z-score normalisation
        mean = cube.mean(axis=(1, 2), keepdims=True)
        std  = cube.std(axis=(1, 2), keepdims=True) + 1e-8
        cube = (cube - mean) / std

        # Step 6: To tensor (1, Bands, H, W)
        cube_t = torch.from_numpy(cube).unsqueeze(0)  # add channel dim

        health_label, contam_label = self.labels[idx]
        return (
            cube_t,
            torch.tensor(health_label, dtype=torch.long),
            torch.tensor(contam_label, dtype=torch.float32),
        )


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, os

    # Create a tiny fake dataset
    tmp = tempfile.mkdtemp()
    paths, lbls = [], []
    for i in range(6):
        cube = np.random.rand(210, 70, 70).astype(np.float32)  # intentionally off-size
        p = os.path.join(tmp, f"cube_{i}.npy")
        np.save(p, cube)
        paths.append(p)
        lbls.append((np.random.randint(0, 5), np.random.rand(4).tolist()))

    ds_train = HyperspectralSoilDataset(paths[:4], lbls[:4], train=True)
    ds_val   = HyperspectralSoilDataset(paths[4:], lbls[4:], train=False)

    cube_t, health, contam = ds_train[0]
    print("train cube shape :", cube_t.shape)   # (1, 200, 64, 64)
    print("health label     :", health)
    print("contam label     :", contam)
    print("cube dtype       :", cube_t.dtype)
    print("cube mean ~0     :", cube_t.mean().item())

    cube_v, _, _ = ds_val[0]
    print("val cube shape   :", cube_v.shape)   # (1, 200, 64, 64)
