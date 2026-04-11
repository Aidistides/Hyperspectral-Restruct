"""
Atmospheric correction for drone-based hyperspectral imaging.

Removes atmospheric interference effects including scattering, absorption, and path radiance.
Essential for drone imagery due to low altitude and variable atmospheric conditions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import ndimage, interpolate
from sklearn.linear_model import LinearRegression
import warnings

from .config import AtmosphericConfig


class AtmosphericCorrection:
    """
    Atmospheric correction for drone hyperspectral imagery.
    
    Addresses key atmospheric effects:
    - Path radiance (scattering)
    - Absorption (water vapor, ozone)
    - Adjacency effects
    - Illumination variation
    """
    
    def __init__(self, config: AtmosphericConfig, wavelengths: np.ndarray):
        self.config = config
        self.wavelengths = wavelengths
        self.dark_object_spectrum = None
        self.empirical_line_params = None
        
    def find_dark_objects(self, data: np.ndarray) -> np.ndarray:
        """
        Identify dark objects for empirical line calibration.
        
        Args:
            data: Reflectance cube (H x W x C) or (C x H x W)
            
        Returns:
            Dark object spectrum
        """
        print("  - Finding dark objects for empirical line...")
        
        if len(data.shape) == 3:
            # H x W x C format
            # Use low percentile to find dark pixels
            dark_threshold = np.percentile(data, self.config.dark_object_percentile, axis=2)
            dark_mask = np.all(data <= dark_threshold[:, :, np.newaxis], axis=2)
        else:
            # C x H x W format
            dark_threshold = np.percentile(data, self.config.dark_object_percentile, axis=(1, 2))
            dark_mask = np.all(data <= dark_threshold[:, np.newaxis, np.newaxis], axis=0)
        
        # Extract dark object spectrum
        if np.sum(dark_mask) < 100:  # Need enough dark pixels
            warnings.warn("Insufficient dark pixels found, using global minimum")
            if len(data.shape) == 3:
                self.dark_object_spectrum = np.min(data, axis=(0, 1))
            else:
                self.dark_object_spectrum = np.min(data, axis=(1, 2))
        else:
            if len(data.shape) == 3:
                self.dark_object_spectrum = np.mean(data[dark_mask], axis=0)
            else:
                self.dark_object_spectrum = np.mean(data[:, dark_mask], axis=1)
        
        return self.dark_object_spectrum
    
    def empirical_line_correction(self, data: np.ndarray) -> np.ndarray:
        """
        Apply empirical line atmospheric correction.
        
        Based on the empirical line method where pixel values lie on a line
        in spectral space defined by dark object and bright target spectra.
        
        Args:
            data: Reflectance cube with atmospheric effects
            
        Returns:
            Atmospherically corrected reflectance
        """
        if not self.config.use_empirical_line:
            return data
        
        # Find dark objects if not already done
        if self.dark_object_spectrum is None:
            self.find_dark_objects(data)
        
        # Estimate bright targets (high reflectance)
        if len(data.shape) == 3:
            bright_spectrum = np.percentile(data, 99, axis=(0, 1))
        else:
            bright_spectrum = np.percentile(data, 99, axis=(1, 2))
        
        # Calculate empirical line parameters
        # For each band: corrected = (observed - dark) / (bright - dark)
        dark_diff = bright_spectrum - self.dark_object_spectrum
        dark_diff[dark_diff == 0] = 1e-6  # Avoid division by zero
        
        if len(data.shape) == 3:
            corrected = (data - self.dark_object_spectrum) / dark_diff
        else:
            corrected = (data - self.dark_object_spectrum[:, np.newaxis, np.newaxis]) / dark_diff[:, np.newaxis, np.newaxis]
        
        # Clip to valid range
        corrected = np.clip(corrected, 0, 1)
        
        self.empirical_line_params = {
            'dark_object': self.dark_object_spectrum,
            'bright_target': bright_spectrum,
            'slope': dark_diff
        }
        
        return corrected
    
    def water_vapor_correction(self, data: np.ndarray) -> np.ndarray:
        """
        Correct for water vapor absorption bands.
        
        Args:
            data: Atmospherically corrected reflectance
            
        Returns:
            Water vapor corrected reflectance
        """
        if len(self.config.water_vapor_bands) == 0:
            return data
        
        print("  - Correcting water vapor absorption...")
        
        # Find water vapor absorption bands
        water_bands = []
        for band_center in self.config.water_vapor_bands:
            # Find closest wavelength band
            band_idx = np.argmin(np.abs(self.wavelengths - band_center))
            if np.abs(self.wavelengths[band_idx] - band_center) < 10:  # Within 10nm
                water_bands.append(band_idx)
        
        if len(water_bands) == 0:
            return data
        
        # Interpolate across water vapor bands
        if len(data.shape) == 3:
            corrected = data.copy()
            for band_idx in water_bands:
                if band_idx > 0 and band_idx < data.shape[2] - 1:
                    # Linear interpolation from neighboring bands
                    left_band = max(0, band_idx - 1)
                    right_band = min(data.shape[2] - 1, band_idx + 1)
                    
                    # Weighted interpolation based on wavelength distance
                    left_dist = np.abs(self.wavelengths[left_band] - self.wavelengths[band_idx])
                    right_dist = np.abs(self.wavelengths[right_band] - self.wavelengths[band_idx])
                    total_dist = left_dist + right_dist
                    
                    left_weight = right_dist / total_dist
                    right_weight = left_dist / total_dist
                    
                    corrected[:, :, band_idx] = (left_weight * data[:, :, left_band] + 
                                               right_weight * data[:, :, right_band])
        else:
            corrected = data.copy()
            for band_idx in water_bands:
                if band_idx > 0 and band_idx < data.shape[0] - 1:
                    left_band = max(0, band_idx - 1)
                    right_band = min(data.shape[0] - 1, band_idx + 1)
                    
                    left_dist = np.abs(self.wavelengths[left_band] - self.wavelengths[band_idx])
                    right_dist = np.abs(self.wavelengths[right_band] - self.wavelengths[band_idx])
                    total_dist = left_dist + right_dist
                    
                    left_weight = right_dist / total_dist
                    right_weight = left_dist / total_dist
                    
                    corrected[band_idx, :, :] = (left_weight * data[left_band, :, :] + 
                                               right_weight * data[right_band, :, :])
        
        return corrected
    
    def rayleigh_scattering_correction(self, data: np.ndarray) -> np.ndarray:
        """
        Correct for Rayleigh scattering (wavelength-dependent).
        
        Args:
            data: Water vapor corrected reflectance
            
        Returns:
            Rayleigh corrected reflectance
        """
        if not self.config.rayleigh_correction:
            return data
        
        print("  - Correcting Rayleigh scattering...")
        
        # Rayleigh scattering follows λ^(-4) relationship
        # Calculate scattering factor for each wavelength
        reference_wavelength = 550  # nm (green light)
        scattering_factor = (reference_wavelength / self.wavelengths) ** 4
        
        # Normalize scattering factor
        scattering_factor = scattering_factor / np.mean(scattering_factor)
        
        # Apply correction (reduce scattering effect)
        if len(data.shape) == 3:
            corrected = data / scattering_factor[np.newaxis, np.newaxis, :]
        else:
            corrected = data / scattering_factor[:, np.newaxis, np.newaxis]
        
        return corrected
    
    def adjacency_correction(self, data: np.ndarray, 
                          kernel_size: int = 11) -> np.ndarray:
        """
        Correct for adjacency effects (light scattering between neighboring pixels).
        
        Args:
            data: Rayleigh corrected reflectance
            kernel_size: Size of adjacency correction kernel
            
        Returns:
            Adjacency corrected reflectance
        """
        # Simple adjacency correction using point spread function
        # This is a simplified approach - more sophisticated methods exist
        
        if len(data.shape) == 3:
            corrected = np.zeros_like(data)
            for c in range(data.shape[2]):
                # Estimate adjacency contribution
                blurred = ndimage.gaussian_filter(data[:, :, c], sigma=kernel_size/6)
                adjacency = blurred - data[:, :, c]
                
                # Remove adjacency effect (simplified)
                corrected[:, :, c] = data[:, :, c] - 0.3 * adjacency
        else:
            corrected = np.zeros_like(data)
            for c in range(data.shape[0]):
                blurred = ndimage.gaussian_filter(data[c, :, :], sigma=kernel_size/6)
                adjacency = blurred - data[c, :, :]
                corrected[c, :, :] = data[c, :, :] - 0.3 * adjacency
        
        return corrected
    
    def correct(self, data: np.ndarray) -> np.ndarray:
        """
        Apply complete atmospheric correction pipeline.
        
        Args:
            data: Radiometrically corrected reflectance
            
        Returns:
            Atmospherically corrected reflectance
        """
        print("🌤 Applying atmospheric correction...")
        
        # Step 1: Empirical line correction
        if self.config.atmospheric_model == "empirical_line":
            print("  - Applying empirical line correction...")
            data = self.empirical_line_correction(data)
        
        # Step 2: Water vapor correction
        data = self.water_vapor_correction(data)
        
        # Step 3: Rayleigh scattering correction
        data = self.rayleigh_scattering_correction(data)
        
        # Step 4: Adjacency correction (optional, can be computationally expensive)
        # data = self.adjacency_correction(data)
        
        print("✅ Atmospheric correction completed")
        return data
    
    def get_correction_report(self) -> Dict[str, any]:
        """
        Get report of applied atmospheric corrections.
        
        Returns:
            Dictionary with correction status and parameters
        """
        report = {
            "atmospheric_model": self.config.atmospheric_model,
            "water_vapor_bands": self.config.water_vapor_bands,
            "rayleigh_correction": self.config.rayleigh_correction,
            "aerosol_correction": self.config.aerosol_correction
        }
        
        if self.empirical_line_params is not None:
            report["empirical_line"] = self.empirical_line_params
        
        if self.dark_object_spectrum is not None:
            report["dark_object_spectrum"] = self.dark_object_spectrum.tolist()
        
        return report
