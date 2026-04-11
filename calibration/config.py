"""
Calibration configuration for hyperspectral drone imaging systems.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml


@dataclass
class RadiometricConfig:
    """Configuration for radiometric correction parameters."""
    
    # Sensor-specific parameters
    sensor_name: str = "unknown"
    bit_depth: int = 12
    gain: float = 1.0
    offset: float = 0.0
    
    # Calibration coefficients
    dark_current: Optional[Dict[str, float]] = None
    vignetting_correction: bool = True
    non_uniformity_correction: bool = True
    
    # Conversion parameters
    dn_to_radiance_coeffs: Optional[List[float]] = None
    radiance_to_reflectance_method: str = "empirical_line"  # "empirical_line", "flat_field", "invariant"
    
    # Reference panel parameters
    reference_panel_reflectance: Optional[Dict[str, float]] = None
    reference_panel_location: Optional[str] = None


@dataclass
class AtmosphericConfig:
    """Configuration for atmospheric correction parameters."""
    
    # Atmospheric model
    atmospheric_model: str = "dark_object_subtraction"  # "dark_object", "empirical_line", "radiative_transfer"
    
    # Dark object subtraction
    dark_object_percentile: float = 1.0  # percentile for dark object selection
    
    # Empirical line method
    use_empirical_line: bool = True
    min_wavelength: float = 400.0  # nm
    max_wavelength: float = 1000.0  # nm
    
    # Water vapor absorption bands (to be masked)
    water_vapor_bands: List[float] = field(default_factory=lambda: [940, 1140, 1380])  # nm
    
    # Atmospheric scattering correction
    rayleigh_correction: bool = True
    aerosol_correction: bool = True


@dataclass
class GroundTruthConfig:
    """Configuration for ground-truth calibration parameters."""
    
    # Reference targets
    reference_targets_file: Optional[str] = None
    reference_targets: Optional[Dict[str, Dict[str, float]]] = None  # {target_name: {wavelength: reflectance}}
    
    # Calibration method
    calibration_method: str = "linear_regression"  # "linear_regression", "polynomial", "ratio"
    
    # Quality control
    min_reference_samples: int = 3
    max_calibration_rmse: float = 0.05  # Maximum acceptable RMSE
    outlier_detection: bool = True
    outlier_threshold: float = 2.0  # Standard deviations
    
    # Validation
    cross_validation_folds: int = 5
    calibration_uncertainty_threshold: float = 0.1


@dataclass
class CalibrationConfig:
    """Main configuration class for calibration pipeline."""
    
    radiometric: RadiometricConfig = field(default_factory=RadiometricConfig)
    atmospheric: AtmosphericConfig = field(default_factory=AtmosphericConfig)
    ground_truth: GroundTruthConfig = field(default_factory=GroundTruthConfig)
    
    # Pipeline settings
    enable_radiometric: bool = True
    enable_atmospheric: bool = True
    enable_ground_truth: bool = True
    
    # Output settings
    output_format: str = "reflectance"  # "radiance", "reflectance"
    quality_metrics: bool = True
    save_intermediate: bool = False
    output_directory: str = "calibrated_data"
    
    # Validation
    validate_calibration: bool = True
    calibration_report: bool = True
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'CalibrationConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            radiometric=RadiometricConfig(**config_dict.get('radiometric', {})),
            atmospheric=AtmosphericConfig(**config_dict.get('atmospheric', {})),
            ground_truth=GroundTruthConfig(**config_dict.get('ground_truth', {})),
            **{k: v for k, v in config_dict.items() 
               if k not in ['radiometric', 'atmospheric', 'ground_truth']}
        )
    
    def to_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'radiometric': self.radiometric.__dict__,
            'atmospheric': self.atmospheric.__dict__,
            'ground_truth': self.ground_truth.__dict__,
            'enable_radiometric': self.enable_radiometric,
            'enable_atmospheric': self.enable_atmospheric,
            'enable_ground_truth': self.enable_ground_truth,
            'output_format': self.output_format,
            'quality_metrics': self.quality_metrics,
            'save_intermediate': self.save_intermediate,
            'output_directory': self.output_directory,
            'validate_calibration': self.validate_calibration,
            'calibration_report': self.calibration_report
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        warnings = []
        
        if self.radiometric.bit_depth <= 0:
            warnings.append("Bit depth must be positive")
        
        if self.atmospheric.dark_object_percentile < 0 or self.atmospheric.dark_object_percentile > 100:
            warnings.append("Dark object percentile must be between 0 and 100")
        
        if self.ground_truth.min_reference_samples < 1:
            warnings.append("Minimum reference samples must be at least 1")
        
        if self.ground_truth.max_calibration_rmse < 0:
            warnings.append("Maximum calibration RMSE must be non-negative")
        
        return warnings
