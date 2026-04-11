"""
Radiometric correction for drone-based hyperspectral imaging.

Converts raw digital numbers (DN) to physically meaningful radiance and reflectance values.
Addresses sensor-specific characteristics and provides the foundation for atmospheric correction.
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from scipy import ndimage
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings

from .config import RadiometricConfig


class RadiometricCorrection:
    """
    Radiometric correction for hyperspectral drone imagery.
    
    This class implements the conversion from raw digital numbers to radiance
    and reflectance values, addressing key sensor characteristics:
    - Dark current correction
    - Vignetting correction
    - Non-uniformity correction
    - DN to radiance conversion
    - Radiance to reflectance conversion
    """
    
    def __init__(self, config: RadiometricConfig):
        self.config = config
        self.dark_current_corrected = False
        self.vignetting_corrected = False
        self.non_uniformity_corrected = False
        
    def correct_dark_current(self, data: np.ndarray) -> np.ndarray:
        """
        Remove dark current/sensor noise from raw data.
        
        Args:
            data: Raw hyperspectral cube (H x W x C) or (C x H x W)
            
        Returns:
            Dark current corrected data
        """
        if self.config.dark_current is None:
            # Estimate dark current from darkest pixels
            if len(data.shape) == 3:
                # Assume H x W x C format
                dark_values = np.percentile(data, self.config.dark_current_percentile, axis=(0, 1))
            else:
                # Assume C x H x W format
                dark_values = np.percentile(data, self.config.dark_current_percentile, axis=(1, 2))
        else:
            # Use provided dark current values
            dark_values = np.array([self.config.dark_current.get(f'band_{i}', 0) 
                                  for i in range(data.shape[-1])])
        
        # Apply dark current correction
        if len(data.shape) == 3:
            corrected = data - dark_values
        else:
            corrected = data - dark_values[:, np.newaxis, np.newaxis]
        
        self.dark_current_corrected = True
        return corrected
    
    def correct_vignetting(self, data: np.ndarray) -> np.ndarray:
        """
        Correct vignetting effects (radial intensity fall-off).
        
        Args:
            data: Dark current corrected hyperspectral cube
            
        Returns:
            Vignetting corrected data
        """
        if not self.config.vignetting_correction:
            return data
        
        h, w = data.shape[:2]
        center_y, center_x = h // 2, w // 2
        
        # Create distance map from center
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # Normalize distance to [0, 1]
        normalized_distance = distance / max_distance
        
        # Estimate vignetting function from data
        # Assume vignetting causes radial intensity fall-off
        if len(data.shape) == 3:
            # Use mean across bands to estimate vignetting
            mean_image = np.mean(data, axis=2)
        else:
            mean_image = np.mean(data, axis=0)
        
        # Fit vignetting correction function
        # Simple model: vignetting = a * r^2 + b * r + c
        r = normalized_distance.flatten()
        intensity = mean_image.flatten()
        
        # Remove outliers (bright objects)
        valid_mask = intensity > np.percentile(intensity, 10)
        if np.sum(valid_mask) > 100:  # Need enough valid pixels
            coeffs = np.polyfit(r[valid_mask], intensity[valid_mask], 2)
            vignetting_model = np.polyval(coeffs, r).reshape(h, w)
            
            # Normalize vignetting model
            vignetting_model = vignetting_model / np.max(vignetting_model)
            
            # Apply correction
            if len(data.shape) == 3:
                corrected = data / vignetting_model[:, :, np.newaxis]
            else:
                corrected = data / vignetting_model[np.newaxis, :, :]
        else:
            # Fallback: simple radial correction
            vignetting_model = 1 - 0.3 * normalized_distance**2
            if len(data.shape) == 3:
                corrected = data / vignetting_model[:, :, np.newaxis]
            else:
                corrected = data / vignetting_model[np.newaxis, :, :]
        
        self.vignetting_corrected = True
        return corrected
    
    def correct_non_uniformity(self, data: np.ndarray) -> np.ndarray:
        """
        Correct sensor non-uniformity using flat-field correction.
        
        Args:
            data: Vignetting corrected hyperspectral cube
            
        Returns:
            Non-uniformity corrected data
        """
        if not self.config.non_uniformity_correction:
            return data
        
        # Estimate flat-field from homogeneous areas
        if len(data.shape) == 3:
            # H x W x C format
            flat_field = np.zeros_like(data)
            for c in range(data.shape[2]):
                band = data[:, :, c]
                # Use low-pass filter to estimate illumination pattern
                flat_field[:, :, c] = cv2.GaussianBlur(band, (51, 51), 0)
        else:
            # C x H x W format
            flat_field = np.zeros_like(data)
            for c in range(data.shape[0]):
                band = data[c, :, :]
                flat_field[c, :, :] = cv2.GaussianBlur(band, (51, 51), 0)
        
        # Normalize flat-field
        flat_field = flat_field / np.percentile(flat_field, 95, axis=(-2, -1), keepdims=True)
        
        # Apply correction
        corrected = data / flat_field
        
        self.non_uniformity_corrected = True
        return corrected
    
    def dn_to_radiance(self, data: np.ndarray) -> np.ndarray:
        """
        Convert digital numbers to radiance values.
        
        Args:
            data: Calibrated DN values
            
        Returns:
            Radiance values (W/m²/sr/μm)
        """
        if self.config.dn_to_radiance_coeffs is not None:
            # Use provided calibration coefficients
            coeffs = np.array(self.config.dn_to_radiance_coeffs)
            if len(data.shape) == 3:
                radiance = np.zeros_like(data)
                for c in range(data.shape[2]):
                    radiance[:, :, c] = coeffs[c] * data[:, :, c] + self.config.offset
            else:
                radiance = np.zeros_like(data)
                for c in range(data.shape[0]):
                    radiance[c, :, :] = coeffs[c] * data[c, :, :] + self.config.offset
        else:
            # Simple linear conversion
            radiance = self.config.gain * data + self.config.offset
        
        # Ensure non-negative radiance
        radiance = np.maximum(radiance, 0)
        
        return radiance
    
    def radiance_to_reflectance(self, radiance: np.ndarray, 
                             solar_irradiance: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convert radiance to reflectance using empirical line method.
        
        Args:
            radiance: Radiance values
            solar_irradiance: Solar irradiance spectrum (optional)
            
        Returns:
            Reflectance values (0-1)
        """
        if self.config.radiance_to_reflectance_method == "empirical_line":
            # Empirical line method: ρ = π * L / E₀
            # Where L is radiance, E₀ is solar irradiance
            
            if solar_irradiance is None:
                # Estimate solar irradiance from data
                # Use bright, dark pixels to estimate empirical line
                if len(radiance.shape) == 3:
                    # H x W x C format
                    bright_radiance = np.percentile(radiance, 99, axis=(0, 1))
                    dark_radiance = np.percentile(radiance, 1, axis=(0, 1))
                else:
                    # C x H x W format
                    bright_radiance = np.percentile(radiance, 99, axis=(1, 2))
                    dark_radiance = np.percentile(radiance, 1, axis=(1, 2))
                
                # Assume bright targets have reflectance = 1.0, dark targets = 0.0
                solar_irradiance_est = np.pi * (bright_radiance - dark_radiance)
                reflectance = (radiance - dark_radiance[:, np.newaxis, np.newaxis]) / solar_irradiance_est
            else:
                # Use provided solar irradiance
                reflectance = np.pi * radiance / solar_irradiance
                
        elif self.config.radiance_to_reflectance_method == "flat_field":
            # Flat field calibration using reference panel
            if self.config.reference_panel_reflectance is not None:
                # Scale to known reference reflectance
                panel_reflectance = np.array(list(self.config.reference_panel_reflectance.values()))
                if len(radiance.shape) == 3:
                    panel_radiance = np.mean(radiance, axis=(0, 1))
                else:
                    panel_radiance = np.mean(radiance, axis=(1, 2))
                
                # Linear scaling
                scale_factor = panel_reflectance / panel_radiance
                reflectance = radiance * scale_factor
            else:
                warnings.warn("Reference panel reflectance not provided, using simple normalization")
                reflectance = radiance / np.max(radiance)
                
        else:
            # Simple normalization
            reflectance = radiance / np.max(radiance)
        
        # Ensure reflectance is in valid range [0, 1]
        reflectance = np.clip(reflectance, 0, 1)
        
        return reflectance
    
    def calibrate(self, data: np.ndarray, 
                 solar_irradiance: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply complete radiometric calibration pipeline.
        
        Args:
            data: Raw hyperspectral cube (DN values)
            solar_irradiance: Solar irradiance spectrum (optional)
            
        Returns:
            Calibrated reflectance cube
        """
        print("🔧 Applying radiometric calibration...")
        
        # Step 1: Dark current correction
        if self.config.enable_radiometric:
            print("  - Correcting dark current...")
            data = self.correct_dark_current(data)
        
        # Step 2: Vignetting correction
        if self.config.vignetting_correction:
            print("  - Correcting vignetting...")
            data = self.correct_vignetting(data)
        
        # Step 3: Non-uniformity correction
        if self.config.non_uniformity_correction:
            print("  - Correcting non-uniformity...")
            data = self.correct_non_uniformity(data)
        
        # Step 4: DN to radiance conversion
        print("  - Converting DN to radiance...")
        radiance = self.dn_to_radiance(data)
        
        # Step 5: Radiance to reflectance conversion
        print("  - Converting radiance to reflectance...")
        reflectance = self.radiance_to_reflectance(radiance, solar_irradiance)
        
        print("✅ Radiometric calibration completed")
        return reflectance
    
    def get_calibration_report(self) -> Dict[str, bool]:
        """
        Get report of applied calibration steps.
        
        Returns:
            Dictionary with calibration status
        """
        return {
            "dark_current_corrected": self.dark_current_corrected,
            "vignetting_corrected": self.vignetting_corrected,
            "non_uniformity_corrected": self.non_uniformity_corrected,
            "sensor_name": self.config.sensor_name,
            "bit_depth": self.config.bit_depth,
            "gain": self.config.gain,
            "offset": self.config.offset
        }
