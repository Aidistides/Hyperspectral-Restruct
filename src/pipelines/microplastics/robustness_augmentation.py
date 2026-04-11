"""
Robustness Module for Microplastics HSI Classification
Inspired by Kitahashi et al. (2021) - Analytical Methods
"Development of robust models for rapid classification of microplastic polymer types 
based on near infrared hyperspectral images"

Focus: Make models invariant to particle size, moisture (wet/dry), and biofouling.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import random

class MP RobustnessAugmentor:
    """
    Applies augmentations and corrections inspired by the paper:
    - Simulate wet-filter effects (stronger absorption at longer wavelengths)
    - Particle size variation (weaker signals for smaller particles)
    - Biofouling simulation (add organic-like spectral interference)
    """
    
    def __init__(self, wavelengths: np.ndarray):
        self.wavelengths = wavelengths  # in nm
        # NIR/SWIR water absorption is stronger > ~1400 nm
        self.water_absorption_mask = wavelengths > 1400
    
    def simulate_wet_filter(self, spectra: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """Reduce reflectance more at longer wavelengths (mimics water absorption)."""
        wet_spectra = spectra.copy()
        # Stronger damping at longer NIR wavelengths
        damping = 1.0 - strength * (self.wavelengths / self.wavelengths.max())
        wet_spectra *= damping[None, :]
        return wet_spectra
    
    def simulate_small_particle(self, spectra: np.ndarray, factor: float = 0.6) -> np.ndarray:
        """Weaker overall signal for smaller particles (100-500 μm vs 1mm)."""
        return spectra * factor
    
    def simulate_biofouling(self, spectra: np.ndarray, intensity: float = 0.15) -> np.ndarray:
        """Add organic/biological interference (e.g., microalgae-like absorption)."""
        fouled = spectra.copy()
        # Simple broad absorption in certain bands (can be refined with real organic spectra)
        noise = np.random.normal(0, intensity, spectra.shape)
        fouled += noise * (1 - spectra)  # stronger effect where reflectance is high
        return np.clip(fouled, 0, 1)
    
    def augment_sample(self, spectra: np.ndarray, 
                       apply_wet: bool = True,
                       apply_small: bool = True,
                       apply_fouling: bool = True) -> np.ndarray:
        """Combine augmentations to create robust training samples."""
        aug = spectra.copy()
        
        if apply_wet and random.random() > 0.3:
            aug = self.simulate_wet_filter(aug)
        if apply_small and random.random() > 0.4:
            aug = self.simulate_small_particle(aug)
        if apply_fouling and random.random() > 0.5:
            aug = self.simulate_biofouling(aug)
        
        # Optional: add light Gaussian smoothing to mimic real imaging noise
        aug = gaussian_filter(aug, sigma=1.0, axis=1)
        
        return aug
    
    def augment_dataset(self, X: np.ndarray, y: np.ndarray, n_aug_per_sample: int = 3):
        """Generate augmented versions of the dataset for training robust models."""
        X_aug = []
        y_aug = []
        
        for i in range(len(X)):
            X_aug.append(X[i])
            y_aug.append(y[i])
            
            for _ in range(n_aug_per_sample):
                augmented = self.augment_sample(X[i])
                X_aug.append(augmented)
                y_aug.append(y[i])
        
        return np.array(X_aug), np.array(y_aug)


# Example integration with your existing pipeline
if __name__ == "__main__":
    # Assume you have wavelengths array and spectra from your HSI loader
    # wavelengths = np.load("wavelengths.npy")
    # X_spectra, y_labels = load_your_data()
    
    augmentor = MPRobustnessAugmentor(wavelengths)
    X_aug, y_aug = augmentor.augment_dataset(X_spectra, y_labels)
    
    print(f"Original samples: {len(X_spectra)} → Augmented: {len(X_aug)}")
    print("✅ Robustness augmentations added (wet/dry, size, biofouling)")
