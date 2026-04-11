"""
calibration module — Radiometric and atmospheric correction for drone-based hyperspectral imaging.

This module provides comprehensive calibration pipelines to convert raw digital numbers (DN)
 to physically meaningful reflectance values, addressing key limitations mentioned in README:
 - Atmospheric interference correction
 - Ground-truth calibration using reference panels
 - Radiometric normalization for drone-based systems

Key Components:
- RadiometricCorrection: Convert DN to radiance/reflectance
- AtmosphericCorrection: Remove atmospheric scattering/absorption effects
- GroundTruthCalibration: Calibrate using known reference targets
- CalibrationPipeline: End-to-end calibration workflow
"""

from .radiometric import RadiometricCorrection
from .atmospheric import AtmosphericCorrection
from .ground_truth import GroundTruthCalibration
from .pipeline import CalibrationPipeline
from .config import CalibrationConfig
from .utils import validate_calibration_data, calculate_calibration_metrics

__all__ = [
    "RadiometricCorrection",
    "AtmosphericCorrection", 
    "GroundTruthCalibration",
    "CalibrationPipeline",
    "CalibrationConfig",
    "validate_calibration_data",
    "calculate_calibration_metrics",
]
