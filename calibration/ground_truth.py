"""
Ground-truth calibration for drone-based hyperspectral imaging.

Uses known reference targets with measured reflectance spectra to calibrate
the entire imaging system. Essential for accurate quantitative analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
import warnings

from .config import GroundTruthConfig


class GroundTruthCalibration:
    """
    Ground-truth calibration using reference targets.
    
    Implements robust calibration methods:
    - Linear regression calibration
    - Polynomial calibration
    - Ratio-based calibration
    - Cross-validation for quality assessment
    """
    
    def __init__(self, config: GroundTruthConfig, wavelengths: np.ndarray):
        self.config = config
        self.wavelengths = wavelengths
        self.calibration_models = {}
        self.calibration_metrics = {}
        self.reference_spectra = {}
        
    def load_reference_targets(self, reference_data: Union[str, Dict]) -> None:
        """
        Load reference target spectra.
        
        Args:
            reference_data: Either path to reference file or dictionary of reference spectra
        """
        if isinstance(reference_data, str):
            # Load from file (CSV, JSON, etc.)
            try:
                if reference_data.endswith('.csv'):
                    import pandas as pd
                    df = pd.read_csv(reference_data)
                    # Assume columns: wavelength, target1, target2, ...
                    wavelength_col = df.columns[0]
                    self.reference_wavelengths = df[wavelength_col].values
                    
                    for target_name in df.columns[1:]:
                        self.reference_spectra[target_name] = df[target_name].values
                        
                elif reference_data.endswith('.json'):
                    import json
                    with open(reference_data, 'r') as f:
                        data = json.load(f)
                    self.reference_wavelengths = np.array(data['wavelengths'])
                    self.reference_spectra = data['spectra']
                    
                else:
                    raise ValueError(f"Unsupported file format: {reference_data}")
                    
            except Exception as e:
                raise ValueError(f"Failed to load reference data: {e}")
                
        elif isinstance(reference_data, dict):
            # Use provided dictionary
            self.reference_spectra = reference_data
            # Assume wavelengths are provided separately or use class wavelengths
            self.reference_wavelengths = self.wavelengths
            
        else:
            raise ValueError("reference_data must be file path or dictionary")
        
        # Validate reference data
        if len(self.reference_spectra) < self.config.min_reference_samples:
            raise ValueError(f"Need at least {self.config.min_reference_samples} reference targets")
        
        print(f"  - Loaded {len(self.reference_spectra)} reference targets")
    
    def extract_target_spectra(self, data: np.ndarray, 
                             target_masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Extract spectra from image using provided target masks.
        
        Args:
            data: Hyperspectral cube (H x W x C) or (C x H x W)
            target_masks: Dictionary of target name -> mask (H x W)
            
        Returns:
            Dictionary of extracted spectra
        """
        extracted_spectra = {}
        
        for target_name, mask in target_masks.items():
            if np.sum(mask) == 0:
                warnings.warn(f"No pixels found for target: {target_name}")
                continue
            
            # Extract mean spectrum from masked region
            if len(data.shape) == 3:
                # H x W x C format
                spectrum = np.mean(data[mask], axis=0)
            else:
                # C x H x W format
                spectrum = np.mean(data[:, mask], axis=1)
            
            extracted_spectra[target_name] = spectrum
        
        return extracted_spectra
    
    def linear_regression_calibration(self, 
                                   measured_spectra: Dict[str, np.ndarray],
                                   reference_spectra: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, LinearRegression]:
        """
        Calibrate using linear regression between measured and reference spectra.
        
        Args:
            measured_spectra: Measured spectra from image
            reference_spectra: Reference spectra (optional, uses loaded if None)
            
        Returns:
            Dictionary of calibration models
        """
        if reference_spectra is None:
            reference_spectra = self.reference_spectra
        
        models = {}
        
        # Prepare data for regression
        for target_name in measured_spectra:
            if target_name not in reference_spectra:
                warnings.warn(f"No reference spectrum for target: {target_name}")
                continue
            
            measured = measured_spectra[target_name]
            reference = reference_spectra[target_name]
            
            # Interpolate reference spectrum to match wavelengths
            if len(reference) != len(measured):
                if hasattr(self, 'reference_wavelengths'):
                    ref_interp = np.interp(self.wavelengths, 
                                          self.reference_wavelengths, reference)
                else:
                    warnings.warn(f"Wavelength mismatch for target: {target_name}")
                    continue
            else:
                ref_interp = reference
            
            # Remove outliers
            if self.config.outlier_detection:
                # Simple outlier detection based on residual analysis
                initial_model = LinearRegression()
                initial_model.fit(measured.reshape(-1, 1), ref_interp)
                residuals = ref_interp - initial_model.predict(measured.reshape(-1, 1))
                threshold = self.config.outlier_threshold * np.std(residuals)
                valid_mask = np.abs(residuals) < threshold
                
                if np.sum(valid_mask) < len(measured) * 0.5:  # Keep at least 50% of data
                    warnings.warn(f"Too many outliers detected for target: {target_name}")
                    valid_mask = np.ones(len(measured), dtype=bool)
            else:
                valid_mask = np.ones(len(measured), dtype=bool)
            
            # Fit calibration model
            model = LinearRegression()
            model.fit(measured[valid_mask].reshape(-1, 1), ref_interp[valid_mask])
            
            # Validate model
            predicted = model.predict(measured.reshape(-1, 1))
            mse = mean_squared_error(ref_interp, predicted)
            r2 = r2_score(ref_interp, predicted)
            
            if mse > self.config.max_calibration_rmse ** 2:
                warnings.warn(f"High calibration error for target {target_name}: MSE = {mse:.6f}")
            
            models[target_name] = model
            self.calibration_metrics[target_name] = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2,
                'n_valid': np.sum(valid_mask)
            }
        
        return models
    
    def polynomial_calibration(self, 
                             measured_spectra: Dict[str, np.ndarray],
                             reference_spectra: Optional[Dict[str, np.ndarray]] = None,
                             degree: int = 2) -> Dict[str, Ridge]:
        """
        Calibrate using polynomial regression.
        
        Args:
            measured_spectra: Measured spectra from image
            reference_spectra: Reference spectra (optional)
            degree: Polynomial degree
            
        Returns:
            Dictionary of calibration models
        """
        if reference_spectra is None:
            reference_spectra = self.reference_spectra
        
        models = {}
        
        for target_name in measured_spectra:
            if target_name not in reference_spectra:
                continue
            
            measured = measured_spectra[target_name]
            reference = reference_spectra[target_name]
            
            # Interpolate reference spectrum
            if len(reference) != len(measured):
                if hasattr(self, 'reference_wavelengths'):
                    ref_interp = np.interp(self.wavelengths, 
                                          self.reference_wavelengths, reference)
                else:
                    continue
            else:
                ref_interp = reference
            
            # Create polynomial features
            poly_features = PolynomialFeatures(degree=degree)
            X_poly = poly_features.fit_transform(measured.reshape(-1, 1))
            
            # Fit Ridge regression to prevent overfitting
            model = Ridge(alpha=0.1)
            model.fit(X_poly, ref_interp)
            
            # Validate model
            predicted = model.predict(X_poly)
            mse = mean_squared_error(ref_interp, predicted)
            r2 = r2_score(ref_interp, predicted)
            
            models[target_name] = model
            self.calibration_metrics[target_name] = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2,
                'degree': degree
            }
        
        return models
    
    def ratio_calibration(self, 
                         measured_spectra: Dict[str, np.ndarray],
                         reference_spectra: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        """
        Calibrate using simple ratio method.
        
        Args:
            measured_spectra: Measured spectra from image
            reference_spectra: Reference spectra (optional)
            
        Returns:
            Dictionary of calibration factors
        """
        if reference_spectra is None:
            reference_spectra = self.reference_spectra
        
        calibration_factors = {}
        
        for target_name in measured_spectra:
            if target_name not in reference_spectra:
                continue
            
            measured = measured_spectra[target_name]
            reference = reference_spectra[target_name]
            
            # Interpolate reference spectrum
            if len(reference) != len(measured):
                if hasattr(self, 'reference_wavelengths'):
                    ref_interp = np.interp(self.wavelengths, 
                                          self.reference_wavelengths, reference)
                else:
                    continue
            else:
                ref_interp = reference
            
            # Calculate ratio (avoid division by zero)
            ratio = ref_interp / (measured + 1e-10)
            calibration_factors[target_name] = np.median(ratio)
            
            # Calculate metrics
            calibrated = measured * calibration_factors[target_name]
            mse = mean_squared_error(ref_interp, calibrated)
            r2 = r2_score(ref_interp, calibrated)
            
            self.calibration_metrics[target_name] = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2,
                'calibration_factor': calibration_factors[target_name]
            }
        
        return calibration_factors
    
    def cross_validate_calibration(self, 
                                 measured_spectra: Dict[str, np.ndarray],
                                 reference_spectra: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        """
        Perform cross-validation on calibration models.
        
        Args:
            measured_spectra: Measured spectra from image
            reference_spectra: Reference spectra (optional)
            
        Returns:
            Cross-validation scores
        """
        if reference_spectra is None:
            reference_spectra = self.reference_spectra
        
        cv_scores = {}
        
        for target_name in measured_spectra:
            if target_name not in reference_spectra:
                continue
            
            measured = measured_spectra[target_name]
            reference = reference_spectra[target_name]
            
            # Interpolate reference spectrum
            if len(reference) != len(measured):
                if hasattr(self, 'reference_wavelengths'):
                    ref_interp = np.interp(self.wavelengths, 
                                          self.reference_wavelengths, reference)
                else:
                    continue
            else:
                ref_interp = reference
            
            # Prepare data
            X = measured.reshape(-1, 1)
            y = ref_interp
            
            # Cross-validation
            kf = KFold(n_splits=self.config.cross_validation_folds, shuffle=True, random_state=42)
            
            if self.config.calibration_method == "linear_regression":
                model = LinearRegression()
            elif self.config.calibration_method == "polynomial":
                model = Ridge(alpha=0.1)
                poly_features = PolynomialFeatures(degree=2)
                X = poly_features.fit_transform(X)
            else:
                continue
            
            scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
            cv_scores[target_name] = -np.mean(scores)  # Convert back to MSE
        
        return cv_scores
    
    def apply_calibration(self, data: np.ndarray) -> np.ndarray:
        """
        Apply calibration models to hyperspectral data.
        
        Args:
            data: Hyperspectral cube to calibrate
            
        Returns:
            Calibrated hyperspectral cube
        """
        if not self.calibration_models:
            raise ValueError("No calibration models available. Run calibration first.")
        
        print("  - Applying ground-truth calibration...")
        
        # Use the best calibration model (based on R² score)
        best_target = max(self.calibration_metrics.keys(), 
                          key=lambda x: self.calibration_metrics[x].get('r2', 0))
        best_model = self.calibration_models[best_target]
        
        calibrated = np.zeros_like(data)
        
        if self.config.calibration_method == "ratio":
            # Apply ratio calibration
            calibration_factor = best_model
            if len(data.shape) == 3:
                calibrated = data * calibration_factor
            else:
                calibrated = data * calibration_factor
        else:
            # Apply regression calibration
            if len(data.shape) == 3:
                for c in range(data.shape[2]):
                    band_data = data[:, :, c].reshape(-1, 1)
                    if self.config.calibration_method == "polynomial":
                        poly_features = PolynomialFeatures(degree=2)
                        band_data = poly_features.fit_transform(band_data)
                    calibrated[:, :, c] = best_model.predict(band_data).reshape(data[:, :, c].shape)
            else:
                for c in range(data.shape[0]):
                    band_data = data[c, :, :].reshape(-1, 1)
                    if self.config.calibration_method == "polynomial":
                        poly_features = PolynomialFeatures(degree=2)
                        band_data = poly_features.fit_transform(band_data)
                    calibrated[c, :, :] = best_model.predict(band_data).reshape(data[c, :, :].shape)
        
        return calibrated
    
    def calibrate(self, data: np.ndarray, 
                 target_masks: Optional[Dict[str, np.ndarray]] = None,
                 measured_spectra: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """
        Perform complete ground-truth calibration.
        
        Args:
            data: Hyperspectral cube to calibrate
            target_masks: Optional masks for reference targets
            measured_spectra: Optional pre-extracted measured spectra
            
        Returns:
            Calibrated hyperspectral cube
        """
        print("target Applying ground-truth calibration...")
        
        # Extract measured spectra if not provided
        if measured_spectra is None:
            if target_masks is not None:
                measured_spectra = self.extract_target_spectra(data, target_masks)
            else:
                raise ValueError("Either target_masks or measured_spectra must be provided")
        
        # Perform calibration based on method
        if self.config.calibration_method == "linear_regression":
            self.calibration_models = self.linear_regression_calibration(measured_spectra)
        elif self.config.calibration_method == "polynomial":
            self.calibration_models = self.polynomial_calibration(measured_spectra)
        elif self.config.calibration_method == "ratio":
            self.calibration_models = self.ratio_calibration(measured_spectra)
        else:
            raise ValueError(f"Unknown calibration method: {self.config.calibration_method}")
        
        # Cross-validation
        cv_scores = self.cross_validate_calibration(measured_spectra)
        
        # Apply calibration
        calibrated_data = self.apply_calibration(data)
        
        print("target Ground-truth calibration completed")
        return calibrated_data
    
    def get_calibration_report(self) -> Dict[str, any]:
        """
        Get comprehensive calibration report.
        
        Returns:
            Dictionary with calibration metrics and quality indicators
        """
        report = {
            'calibration_method': self.config.calibration_method,
            'num_reference_targets': len(self.reference_spectra),
            'wavelength_range': [float(np.min(self.wavelengths)), float(np.max(self.wavelengths))],
            'target_metrics': self.calibration_metrics
        }
        
        # Add overall quality metrics
        if self.calibration_metrics:
            rmse_values = [m.get('rmse', 0) for m in self.calibration_metrics.values()]
            r2_values = [m.get('r2', 0) for m in self.calibration_metrics.values()]
            
            report['overall_rmse'] = np.mean(rmse_values)
            report['overall_r2'] = np.mean(r2_values)
            report['calibration_quality'] = 'good' if np.mean(r2_values) > 0.9 else 'fair' if np.mean(r2_values) > 0.7 else 'poor'
        
        return report
