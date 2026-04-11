#!/usr/bin/env python3
"""
Example usage of the hyperspectral calibration pipeline.

This script demonstrates how to use the calibration module for
drone-based hyperspectral imaging systems.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from calibration import CalibrationPipeline, CalibrationConfig
from calibration.utils import create_synthetic_reference_targets, estimate_solar_irradiance


def create_example_data():
    """Create example hyperspectral data for demonstration."""
    # Create synthetic wavelengths (400-1000nm, 5nm steps)
    wavelengths = np.arange(400, 1005, 5)
    
    # Create synthetic hyperspectral cube (100x100 pixels, 121 bands)
    height, width = 100, 100
    num_bands = len(wavelengths)
    
    # Generate synthetic scene with different materials
    data = np.zeros((height, width, num_bands))
    
    # Add vegetation (top-left quadrant)
    vegetation_mask = np.zeros((height, width), dtype=bool)
    vegetation_mask[:50, :50] = True
    
    # Add soil (bottom-right quadrant)
    soil_mask = np.zeros((height, width), dtype=bool)
    soil_mask[50:, 50:] = True
    
    # Add water (small region in center)
    water_mask = np.zeros((height, width), dtype=bool)
    water_mask[45:55, 45:55] = True
    
    # Create reference spectra
    reference_targets = create_synthetic_reference_targets(wavelengths)
    
    # Fill data with reference spectra plus noise
    for i in range(height):
        for j in range(width):
            if vegetation_mask[i, j]:
                spectrum = reference_targets['vegetation']
            elif soil_mask[i, j]:
                spectrum = reference_targets['soil']
            elif water_mask[i, j]:
                spectrum = reference_targets['water']
            else:
                # Mixed pixels
                spectrum = 0.5 * reference_targets['soil'] + 0.3 * reference_targets['vegetation']
            
            # Add sensor noise and atmospheric effects
            noise = np.random.normal(0, 0.02, num_bands)
            atmospheric_effect = 0.1 * np.exp(-((wavelengths - 550) / 200) ** 2)
            
            data[i, j, :] = spectrum * (1 + atmospheric_effect) + noise
    
    # Convert to digital numbers (simulate 12-bit sensor)
    dn_data = np.clip(data * 4095, 0, 4095).astype(np.uint16)
    
    return dn_data, wavelengths, vegetation_mask, soil_mask, water_mask, reference_targets


def create_target_masks(vegetation_mask, soil_mask, water_mask):
    """Create target masks for ground-truth calibration."""
    # Create smaller masks for calibration targets (avoid edge effects)
    calibration_masks = {}
    
    # Vegetation calibration target (center of vegetation area)
    veg_cal_mask = np.zeros_like(vegetation_mask)
    veg_cal_mask[20:30, 20:30] = vegetation_mask[20:30, 20:30]
    calibration_masks['vegetation'] = veg_cal_mask
    
    # Soil calibration target (center of soil area)
    soil_cal_mask = np.zeros_like(soil_mask)
    soil_cal_mask[60:70, 60:70] = soil_mask[60:70, 60:70]
    calibration_masks['soil'] = soil_cal_mask
    
    # Water calibration target (center of water area)
    water_cal_mask = np.zeros_like(water_mask)
    water_cal_mask[47:53, 47:53] = water_mask[47:53, 47:53]
    calibration_masks['water'] = water_cal_mask
    
    return calibration_masks


def main():
    """Main example function."""
    print("Hyperspectral Calibration Pipeline Example")
    print("=" * 50)
    
    # Create example data
    print("Creating example data...")
    dn_data, wavelengths, veg_mask, soil_mask, water_mask, reference_targets = create_example_data()
    target_masks = create_target_masks(veg_mask, soil_mask, water_mask)
    
    print(f"Data shape: {dn_data.shape}")
    print(f"Wavelength range: {wavelengths[0]}-{wavelengths[-1]} nm")
    print(f"Number of bands: {len(wavelengths)}")
    
    # Load calibration configuration
    print("\nLoading calibration configuration...")
    config_path = Path(__file__).parent / "configs" / "default.yaml"
    config = CalibrationConfig.from_yaml(str(config_path))
    
    # Validate configuration
    warnings = config.validate()
    if warnings:
        print("Configuration warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    # Initialize calibration pipeline
    print("\nInitializing calibration pipeline...")
    pipeline = CalibrationPipeline(config, wavelengths)
    
    # Estimate solar irradiance (in practice, use measured data)
    solar_irradiance = estimate_solar_irradiance(wavelengths)
    
    # Run calibration
    print("\nRunning calibration pipeline...")
    calibrated_data, calibration_report = pipeline.calibrate_single_cube(
        dn_data,
        target_masks=target_masks,
        reference_data=reference_targets,
        solar_irradiance=solar_irradiance,
        save_intermediate=True
    )
    
    # Display calibration results
    print("\nCalibration Results:")
    print(f"Processing time: {calibration_report['total_processing_time']:.2f} seconds")
    print(f"Completed steps: {[step['step'] for step in calibration_report['pipeline_steps']]}")
    
    if 'quality_metrics' in calibration_report:
        quality = calibration_report['quality_metrics']
        if 'statistical_metrics' in quality:
            orig_stats = quality['statistical_metrics']['original_data']
            calib_stats = quality['statistical_metrics']['calibrated_data']
            print(f"Data range: {orig_stats['min']:.2f}-{orig_stats['max']:.2f} -> {calib_stats['min']:.3f}-{calib_stats['max']:.3f}")
    
    # Generate calibration report
    print("\nGenerating calibration report...")
    output_dir = Path("calibration_output")
    output_dir.mkdir(exist_ok=True)
    
    report = pipeline.generate_calibration_report(str(output_dir / "calibration_report.json"))
    
    # Save intermediate results
    pipeline.save_intermediate_results(str(output_dir / "intermediate"))
    
    # Save calibrated data
    np.save(output_dir / "calibrated_data.npy", calibrated_data)
    
    # Visualize results (optional)
    try:
        visualize_results(dn_data, calibrated_data, wavelengths, target_masks)
    except ImportError:
        print("Matplotlib not available, skipping visualization")
    
    print(f"\nCalibration completed successfully!")
    print(f"Results saved to: {output_dir}")
    
    return calibrated_data, calibration_report


def visualize_results(original_data, calibrated_data, wavelengths, target_masks):
    """Visualize calibration results."""
    print("Creating visualization...")
    
    # Select representative bands for visualization
    band_indices = {
        'blue': np.argmin(np.abs(wavelengths - 450)),
        'green': np.argmin(np.abs(wavelengths - 550)),
        'red': np.argmin(np.argmin(np.abs(wavelengths - 670))),
        'nir': np.argmin(np.abs(wavelengths - 800))
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original data
    for i, (color, band_idx) in enumerate(band_indices.items()):
        axes[0, i].imshow(original_data[:, :, band_idx], cmap='gray')
        axes[0, i].set_title(f'Original {color.title()} ({wavelengths[band_idx]}nm)')
        axes[0, i].axis('off')
    
    # Calibrated data
    for i, (color, band_idx) in enumerate(band_indices.items()):
        axes[1, i].imshow(calibrated_data[:, :, band_idx], cmap='gray')
        axes[1, i].set_title(f'Calibrated {color.title()} ({wavelengths[band_idx]}nm)')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('calibration_output/calibration_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot spectra
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mean spectra comparison
    original_mean = np.mean(original_data, axis=(0, 1))
    calibrated_mean = np.mean(calibrated_data, axis=(0, 1))
    
    axes[0].plot(wavelengths, original_mean / np.max(original_mean), 'b-', label='Original (normalized)', alpha=0.7)
    axes[0].plot(wavelengths, calibrated_mean, 'r-', label='Calibrated', linewidth=2)
    axes[0].set_xlabel('Wavelength (nm)')
    axes[0].set_ylabel('Reflectance')
    axes[0].set_title('Mean Spectra Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Target spectra
    for target_name, mask in target_masks.items():
        if np.sum(mask) > 0:
            target_spectrum = np.mean(calibrated_data[mask], axis=0)
            axes[1].plot(wavelengths, target_spectrum, label=f'{target_name}', linewidth=2)
    
    axes[1].set_xlabel('Wavelength (nm)')
    axes[1].set_ylabel('Reflectance')
    axes[1].set_title('Calibration Target Spectra')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('calibration_output/spectral_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Visualization saved to calibration_output/")


if __name__ == "__main__":
    main()
