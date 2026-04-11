# New Datasets Setup Guide

This guide covers the setup and usage of the newly added hyperspectral soil datasets.

## Quick Start

```bash
# List all available datasets
python download_datasets.py --list

# Download a specific dataset
python download_datasets.py --dataset munsell

# Download all datasets
python download_datasets.py --dataset all
```

## Dataset Details

### 1. Munsell Soil Color Chart Hyperspectral Dataset

**Best quick win for soil color calibration**

- **Source**: SPECIM IQ camera
- **Spectral Range**: 204 bands (397–1003 nm)
- **Data Types**: 
  - 20×20 voxel chips (~68 MB)
  - Full scenes (~2.1 GB - requires Git LFS)
  - Endmember spectral libraries (~328 KB)
- **License**: CC license via Zenodo
- **DOI**: 10.5281/zenodo.8143355

**Use Cases**:
- Soil color calibration and validation
- Classification benchmarks
- Algorithm testing
- Spectral library development

**Setup**:
```bash
python download_datasets.py --dataset munsell
```

### 2. Database of Hyperspectral Images of Phosphorus in Soil

**Ideal for soil chemistry & nutrient quantification**

- **Source**: Bayspec OCIF push-broom sensor
- **Spectral Range**: 145 bands (420–1000 nm)
- **Samples**: 152 lab-prepared soil samples
- **Data**: Full hyperspectral cubes + chemical reference data
- **Total Size**: ~3 GB across multiple ZIPs
- **Version**: 3 (Mendeley Data)

**Use Cases**:
- Phosphorus quantification
- Nutrient analysis
- Soil chemistry modeling
- Lab-to-field transfer learning

**Setup**:
```bash
python download_datasets.py --dataset phosphorus
```

### 3. Indian Pines AVIRIS Dataset (Site 3)

**Field-scale agriculture and soil mapping**

- **Source**: AVIRIS airborne sensor
- **Spectral Range**: 220 bands (400–2500 nm)
- **Coverage**: ~2-mile × 2-mile area
- **Resolution**: ~20 meters
- **Format**: Native multi-band GeoTIFF
- **Location**: Purdue Agronomy farm

**Use Cases**:
- Large-scale soil mapping
- Agricultural residue studies
- Remote sensing workflow validation
- Field-scale analysis

**Setup**:
```bash
python download_datasets.py --dataset indian_pines
```

## Integration with Existing Pipeline

### Model Configuration Updates

Update your `configs/default.yaml` to handle different spectral ranges:

```yaml
model:
  # For Munsell dataset
  num_bands: 204
  target_size: [32, 32]
  
  # For Phosphorus dataset  
  num_bands: 145
  target_size: [32, 32]
  
  # For Indian Pines
  num_bands: 220
  target_size: [32, 32]
```

### Data Loading

The datasets use different formats. Use the appropriate data loader:

```python
# For Munsell chips (numpy arrays)
from dataset import HyperspectralSoilDataset

# For Indian Pines (GeoTIFF)
from rasterio import open as rio_open

# For Phosphorus (custom format)
# See dataset-specific loading functions
```

### Training Pipeline

1. **Start with Munsell** for color calibration baseline
2. **Add Phosphorus** for chemistry-specific features
3. **Use Indian Pines** for large-scale validation
4. **Fine-tune** on your specific UAV data

## Git LFS Setup

For large files (especially Munsell full scenes):

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.zip"
git lfs track "*.tif"
git lfs track "*.dat"

# Apply tracking
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

## Data Preprocessing

### Spectral Resampling

Different datasets have different spectral ranges. Use spectral resampling:

```python
import numpy as np
from scipy.interpolate import interp1d

def resample_spectrum(wavelengths_orig, spectra_orig, wavelengths_new):
    """Resample spectrum to new wavelength grid."""
    interp_func = interp1d(wavelengths_orig, spectra_orig, kind='linear', 
                          bounds_error=False, fill_value=0)
    return interp_func(wavelengths_new)
```

### Data Augmentation

For the smaller datasets (especially Munsell chips):

```python
# Use existing augmentation in dataset.py
# Consider additional spectral augmentation:
# - Spectral noise injection
# - Wavelength shift simulation
# - Band dropout
```

## Performance Benchmarks

Expected performance metrics for each dataset:

| Dataset | Task | Expected Accuracy | Notes |
|---------|------|-------------------|-------|
| Munsell | Color classification | 85-92% | Excellent color discrimination |
| Phosphorus | Nutrient regression | R²: 0.75-0.85 | Chemical quantification |
| Indian Pines | Land cover classification | 80-88% | Multi-class agricultural |

## Troubleshooting

### Common Issues

1. **Memory errors with large scenes**
   - Use patch-based processing
   - Reduce batch size
   - Enable gradient checkpointing

2. **Spectral mismatch**
   - Resample to common wavelength grid
   - Use spectral alignment techniques
   - Consider spectral attention mechanisms

3. **Download failures**
   - Check internet connection
   - Verify URLs are current
   - Use manual download for large files

### Getting Help

- Check the [Issues](../../issues) page
- Review dataset documentation
- Join the community discussions

## Citation

When using these datasets, please cite:

```bibtex
@dataset{munsell_soil_color_2024,
  author={Various},
  title={Munsell Soil Color Chart Hyperspectral Dataset},
  year={2024},
  publisher={Zenodo},
  doi={10.5281/zenodo.8143355}
}

@dataset{phosphorus_soil_2024,
  author={Various},
  title={Database of Hyperspectral Images of Phosphorus in Soil},
  year={2024},
  publisher={Mendeley Data}
}

@dataset{indian_pines_2024,
  author={Various},
  title={Indian Pines AVIRIS Dataset Site 3},
  year={2024},
  publisher={Purdue University Research Repository}
}
```
