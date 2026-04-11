"""
End-to-end calibration pipeline for drone-based hyperspectral imaging.

Integrates radiometric, atmospheric, and ground-truth calibration into a unified
workflow suitable for production deployment and research applications.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import time
from datetime import datetime

from .config import CalibrationConfig
from .radiometric import RadiometricCorrection
from .atmospheric import AtmosphericCorrection
from .ground_truth import GroundTruthCalibration
from .utils import validate_calibration_data, calculate_calibration_metrics


class CalibrationPipeline:
    """
    Complete calibration pipeline for hyperspectral drone imagery.
    
    This class provides a unified interface for:
    - Radiometric correction (DN to reflectance)
    - Atmospheric interference removal
    - Ground-truth calibration using reference targets
    - Quality assessment and reporting
    - Batch processing capabilities
    """
    
    def __init__(self, config: CalibrationConfig, wavelengths: np.ndarray):
        self.config = config
        self.wavelengths = wavelengths
        self.pipeline_start_time = None
        
        # Initialize calibration components
        if config.enable_radiometric:
            self.radiometric_corrector = RadiometricCorrection(config.radiometric)
        else:
            self.radiometric_corrector = None
            
        if config.enable_atmospheric:
            self.atmospheric_corrector = AtmosphericCorrection(config.atmospheric, wavelengths)
        else:
            self.atmospheric_corrector = None
            
        if config.enable_ground_truth:
            self.ground_truth_calibrator = GroundTruthCalibration(config.ground_truth, wavelengths)
        else:
            self.ground_truth_calibrator = None
        
        # Pipeline state
        self.calibration_history = []
        self.quality_metrics = {}
        self.intermediate_results = {}
        
    def validate_input(self, data: np.ndarray) -> None:
        """
        Validate input hyperspectral data.
        
        Args:
            data: Input hyperspectral cube
            
        Raises:
            ValueError: If data is invalid
        """
        if len(data.shape) != 3:
            raise ValueError(f"Expected 3D data (H x W x C), got shape {data.shape}")
        
        expected_bands = len(self.wavelengths)
        if data.shape[2] != expected_bands:
            raise ValueError(f"Expected {expected_bands} bands, got {data.shape[2]}")
        
        # Check for valid data range
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValueError("Data contains NaN or infinite values")
        
        if np.all(data == 0):
            raise ValueError("Data contains only zeros")
    
    def calibrate_single_cube(self, data: np.ndarray,
                            target_masks: Optional[Dict[str, np.ndarray]] = None,
                            reference_data: Optional[Union[str, Dict]] = None,
                            solar_irradiance: Optional[np.ndarray] = None,
                            save_intermediate: bool = None) -> Tuple[np.ndarray, Dict]:
        """
        Calibrate a single hyperspectral cube.
        
        Args:
            data: Input hyperspectral cube (H x W x C)
            target_masks: Optional masks for ground-truth targets
            reference_data: Optional reference spectra data
            solar_irradiance: Optional solar irradiance spectrum
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Tuple of (calibrated_data, calibration_report)
        """
        if save_intermediate is None:
            save_intermediate = self.config.save_intermediate
        
        self.pipeline_start_time = time.time()
        
        # Validate input
        self.validate_input(data)
        
        # Initialize calibration report
        report = {
            'timestamp': datetime.now().isoformat(),
            'input_shape': data.shape,
            'wavelengths': self.wavelengths.tolist(),
            'pipeline_steps': [],
            'quality_metrics': {},
            'processing_time': {}
        }
        
        current_data = data.copy()
        step_start_time = time.time()
        
        # Step 1: Radiometric correction
        if self.config.enable_radiometric and self.radiometric_corrector is not None:
            print("Step 1: Radiometric correction")
            step_start = time.time()
            
            current_data = self.radiometric_corrector.calibrate(current_data, solar_irradiance)
            
            step_time = time.time() - step_start
            report['pipeline_steps'].append({
                'step': 'radiometric_correction',
                'duration': step_time,
                'status': 'completed'
            })
            report['processing_time']['radiometric'] = step_time
            
            if save_intermediate:
                self.intermediate_results['radiometric'] = current_data.copy()
            
            radiometric_report = self.radiometric_corrector.get_calibration_report()
            report['radiometric_report'] = radiometric_report
        
        # Step 2: Atmospheric correction
        if self.config.enable_atmospheric and self.atmospheric_corrector is not None:
            print("Step 2: Atmospheric correction")
            step_start = time.time()
            
            current_data = self.atmospheric_corrector.correct(current_data)
            
            step_time = time.time() - step_start
            report['pipeline_steps'].append({
                'step': 'atmospheric_correction',
                'duration': step_time,
                'status': 'completed'
            })
            report['processing_time']['atmospheric'] = step_time
            
            if save_intermediate:
                self.intermediate_results['atmospheric'] = current_data.copy()
            
            atmospheric_report = self.atmospheric_corrector.get_correction_report()
            report['atmospheric_report'] = atmospheric_report
        
        # Step 3: Ground-truth calibration
        if self.config.enable_ground_truth and self.ground_truth_calibrator is not None:
            print("Step 3: Ground-truth calibration")
            step_start = time.time()
            
            # Load reference data if provided
            if reference_data is not None:
                self.ground_truth_calibrator.load_reference_targets(reference_data)
            
            # Apply ground-truth calibration
            current_data = self.ground_truth_calibrator.calibrate(
                current_data, target_masks=target_masks
            )
            
            step_time = time.time() - step_start
            report['pipeline_steps'].append({
                'step': 'ground_truth_calibration',
                'duration': step_time,
                'status': 'completed'
            })
            report['processing_time']['ground_truth'] = step_time
            
            if save_intermediate:
                self.intermediate_results['ground_truth'] = current_data.copy()
            
            ground_truth_report = self.ground_truth_calibrator.get_calibration_report()
            report['ground_truth_report'] = ground_truth_report
        
        # Calculate quality metrics
        if self.config.quality_metrics:
            print("Calculating quality metrics...")
            quality_metrics = calculate_calibration_metrics(data, current_data, self.wavelengths)
            report['quality_metrics'] = quality_metrics
            self.quality_metrics = quality_metrics
        
        # Total processing time
        total_time = time.time() - self.pipeline_start_time
        report['total_processing_time'] = total_time
        
        # Validate output
        if self.config.validate_calibration:
            validation_results = validate_calibration_data(current_data, self.wavelengths)
            report['validation_results'] = validation_results
            
            if not validation_results['is_valid']:
                warnings.warn("Calibration validation failed - check quality metrics")
        
        # Add to calibration history
        self.calibration_history.append(report)
        
        print(f"Calibration completed in {total_time:.2f} seconds")
        
        return current_data, report
    
    def calibrate_batch(self, data_list: List[np.ndarray],
                      target_masks_list: Optional[List[Dict[str, np.ndarray]]] = None,
                      reference_data: Optional[Union[str, Dict]] = None,
                      solar_irradiance: Optional[np.ndarray] = None,
                      output_dir: Optional[str] = None) -> List[Tuple[np.ndarray, Dict]]:
        """
        Calibrate multiple hyperspectral cubes in batch.
        
        Args:
            data_list: List of input hyperspectral cubes
            target_masks_list: List of target masks dictionaries
            reference_data: Reference spectra data
            solar_irradiance: Solar irradiance spectrum
            output_dir: Output directory for results
            
        Returns:
            List of (calibrated_data, calibration_report) tuples
        """
        print(f"Starting batch calibration of {len(data_list)} cubes...")
        
        if target_masks_list is not None and len(target_masks_list) != len(data_list):
            raise ValueError("target_masks_list must have same length as data_list")
        
        results = []
        
        for i, data in enumerate(data_list):
            print(f"\nProcessing cube {i+1}/{len(data_list)}...")
            
            target_masks = target_masks_list[i] if target_masks_list is not None else None
            
            try:
                calibrated_data, report = self.calibrate_single_cube(
                    data, target_masks, reference_data, solar_irradiance
                )
                
                results.append((calibrated_data, report))
                
                # Save results if output directory specified
                if output_dir is not None:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    # Save calibrated data
                    data_filename = f"calibrated_cube_{i+1}.npy"
                    np.save(output_path / data_filename, calibrated_data)
                    
                    # Save calibration report
                    report_filename = f"calibration_report_{i+1}.json"
                    with open(output_path / report_filename, 'w') as f:
                        json.dump(report, f, indent=2, default=str)
                
            except Exception as e:
                print(f"Error processing cube {i+1}: {e}")
                # Add error report
                error_report = {
                    'timestamp': datetime.now().isoformat(),
                    'cube_index': i,
                    'error': str(e),
                    'status': 'failed'
                }
                results.append((None, error_report))
        
        # Generate batch summary
        successful_calibrations = sum(1 for data, report in results if data is not None)
        print(f"\nBatch calibration completed: {successful_calibrations}/{len(data_list)} successful")
        
        return results
    
    def generate_calibration_report(self, output_path: Optional[str] = None) -> Dict:
        """
        Generate comprehensive calibration report.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Calibration report dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'pipeline_config': {
                'enable_radiometric': self.config.enable_radiometric,
                'enable_atmospheric': self.config.enable_atmospheric,
                'enable_ground_truth': self.config.enable_ground_truth,
                'output_format': self.config.output_format,
                'validate_calibration': self.config.validate_calibration
            },
            'wavelengths': self.wavelengths.tolist(),
            'total_calibrations': len(self.calibration_history),
            'calibration_history': self.calibration_history,
            'overall_quality_metrics': self.quality_metrics
        }
        
        # Add summary statistics
        if self.calibration_history:
            processing_times = [r.get('total_processing_time', 0) for r in self.calibration_history]
            report['summary_statistics'] = {
                'average_processing_time': np.mean(processing_times),
                'total_processing_time': np.sum(processing_times),
                'success_rate': sum(1 for r in self.calibration_history 
                                   if r.get('status', 'completed') == 'completed') / len(self.calibration_history)
            }
        
        # Save report if path provided
        if output_path is not None:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Calibration report saved to: {output_path}")
        
        return report
    
    def save_intermediate_results(self, output_dir: str) -> None:
        """
        Save intermediate calibration results.
        
        Args:
            output_dir: Directory to save intermediate results
        """
        if not self.intermediate_results:
            print("No intermediate results to save")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for step_name, data in self.intermediate_results.items():
            filename = f"{step_name}_result.npy"
            np.save(output_path / filename, data)
        
        print(f"Intermediate results saved to: {output_dir}")
    
    def get_pipeline_status(self) -> Dict:
        """
        Get current pipeline status and configuration.
        
        Returns:
            Pipeline status dictionary
        """
        return {
            'config': self.config,
            'wavelengths': self.wavelengths.tolist(),
            'components': {
                'radiometric': self.radiometric_corrector is not None,
                'atmospheric': self.atmospheric_corrector is not None,
                'ground_truth': self.ground_truth_calibrator is not None
            },
            'calibration_count': len(self.calibration_history),
            'last_calibration': self.calibration_history[-1] if self.calibration_history else None
        }
