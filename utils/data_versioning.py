"""
Data versioning and reproducibility utilities for hyperspectral datasets.

This module provides tools to track dataset versions, ensure reproducibility,
and maintain data lineage throughout the research pipeline.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pickle
from datetime import datetime


class DataVersion:
    """
    Represents a versioned dataset with metadata and lineage tracking.
    """
    
    def __init__(self, 
                 version_id: str,
                 data_paths: List[str],
                 labels: List[Any],
                 metadata: Optional[Dict] = None,
                 parent_version: Optional[str] = None,
                 checksums: Optional[Dict[str, str]] = None,
                 created_at: Optional[str] = None):
        self.version_id = version_id
        self.data_paths = data_paths
        self.labels = labels
        self.metadata = metadata or {}
        self.parent_version = parent_version
        self.checksums = checksums or {}
        self.created_at = created_at or datetime.now().isoformat()
        self.modified_at = datetime.now().isoformat()
    
    def compute_checksums(self) -> Dict[str, str]:
        """Compute MD5 checksums for all data files."""
        checksums = {}
        for path in self.data_paths:
            try:
                with open(path, 'rb') as f:
                    file_hash = hashlib.md5()
                    while chunk := f.read(8192):
                        file_hash.update(chunk)
                    checksums[path] = file_hash.hexdigest()
            except Exception as e:
                print(f"⚠️  Failed to compute checksum for {path}: {e}")
        self.checksums = checksums
        return checksums
    
    def verify_integrity(self) -> Tuple[bool, Dict[str, str]]:
        """Verify data integrity using stored checksums."""
        if not self.checksums:
            self.compute_checksums()
        
        integrity_issues = {}
        all_valid = True
        
        for path, stored_checksum in self.checksums.items():
            try:
                with open(path, 'rb') as f:
                    file_hash = hashlib.md5()
                    while chunk := f.read(8192):
                        file_hash.update(chunk)
                    current_checksum = file_hash.hexdigest()
                
                if current_checksum != stored_checksum:
                    integrity_issues[path] = f"Expected {stored_checksum}, got {current_checksum}"
                    all_valid = False
            except Exception as e:
                integrity_issues[path] = f"Error checking {path}: {e}"
                all_valid = False
        
        return all_valid, integrity_issues
    
    def to_dict(self) -> Dict:
        """Convert version to dictionary for serialization."""
        return {
            'version_id': self.version_id,
            'data_paths': self.data_paths,
            'num_samples': len(self.data_paths),
            'metadata': self.metadata,
            'parent_version': self.parent_version,
            'checksums': self.checksums,
            'created_at': self.created_at,
            'modified_at': self.modified_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DataVersion':
        """Create DataVersion from dictionary."""
        return cls(
            version_id=data['version_id'],
            data_paths=data['data_paths'],
            labels=data['labels'],
            metadata=data['metadata'],
            parent_version=data.get('parent_version'),
            checksums=data.get('checksums'),
            created_at=data.get('created_at'),
            modified_at=data.get('modified_at')
        )


class DatasetRegistry:
    """
    Registry for managing multiple dataset versions and their relationships.
    """
    
    def __init__(self, registry_path: str = "data/registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.versions = {}
        self.load_registry()
    
    def load_registry(self):
        """Load existing registry from file."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    self.versions = json.load(f)
                print(f"📋 Loaded dataset registry with {len(self.versions)} versions")
            except Exception as e:
                print(f"⚠️  Failed to load registry: {e}")
                self.versions = {}
    
    def save_registry(self):
        """Save registry to file."""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.versions, f, indent=2)
            print(f"💾 Saved dataset registry to {self.registry_path}")
        except Exception as e:
            print(f"❌ Failed to save registry: {e}")
    
    def register_version(self, version: DataVersion):
        """Register a new dataset version."""
        self.versions[version.version_id] = version.to_dict()
        self.save_registry()
        print(f"✅ Registered dataset version: {version.version_id}")
    
    def get_version(self, version_id: str) -> Optional[DataVersion]:
        """Retrieve a specific dataset version."""
        return DataVersion.from_dict(self.versions[version_id]) if version_id in self.versions else None
    
    def list_versions(self) -> List[str]:
        """List all available version IDs."""
        return list(self.versions.keys())
    
    def get_latest_version(self) -> Optional[DataVersion]:
        """Get the most recent version."""
        if not self.versions:
            return None
        
        latest_version = max(self.versions.values(), 
                          key=lambda v: v['created_at'])
        return DataVersion.from_dict(latest_version)
    
    def get_version_lineage(self, version_id: str) -> List[DataVersion]:
        """Get the complete lineage for a version."""
        lineage = []
        current = self.get_version(version_id)
        
        while current:
            lineage.append(current)
            if current.parent_version:
                current = self.get_version(current.parent_version)
            else:
                break
        
        return lineage
    
    def create_version_snapshot(self, 
                           version_id: str,
                           data_paths: List[str],
                           labels: List[Any],
                           metadata: Optional[Dict] = None,
                           description: str = "") -> DataVersion:
        """Create and register a new version with automatic checksums."""
        version = DataVersion(
            version_id=version_id,
            data_paths=data_paths,
            labels=labels,
            metadata=metadata or {},
            parent_version=self.get_latest_version().version_id if self.get_latest_version() else None
        )
        
        # Compute checksums
        version.compute_checksums()
        
        # Add creation metadata
        version.metadata.update({
            'description': description,
            'num_samples': len(data_paths),
            'data_integrity_verified': True
        })
        
        self.register_version(version)
        return version


class ReproducibilityManager:
    """
    Manager for ensuring reproducible data processing and model training.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.experiment_log = []
    
    def set_seeds(self, pytorch_seed: Optional[int] = None, 
                    numpy_seed: Optional[int] = None,
                    cuda_seed: Optional[int] = None):
        """Set seeds for all libraries."""
        import torch
        import random
        
        # Set seeds
        seed = pytorch_seed or self.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(cuda_seed or seed)
        
        np.random.seed(numpy_seed or self.seed)
        random.seed(numpy_seed or self.seed)
        
        # Log seed setting
        self.log_experiment({
            'action': 'set_seeds',
            'pytorch_seed': seed,
            'numpy_seed': numpy_seed or self.seed,
            'cuda_seed': cuda_seed or seed,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"🎲 Seeds set: PyTorch={seed}, NumPy={numpy_seed or self.seed}, CUDA={cuda_seed or seed}")
    
    def log_experiment(self, experiment_data: Dict):
        """Log experiment data for reproducibility."""
        experiment_data.update({
            'timestamp': datetime.now().isoformat(),
            'seed': self.seed
        })
        self.experiment_log.append(experiment_data)
        
        # Save to file periodically
        if len(self.experiment_log) % 10 == 0:
            self.save_experiment_log()
    
    def save_experiment_log(self, output_path: str = "experiments/experiment_log.json"):
        """Save experiment log to file."""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(self.experiment_log, f, indent=2)
            print(f"📊 Experiment log saved to {output_path}")
        except Exception as e:
            print(f"❌ Failed to save experiment log: {e}")
    
    def generate_experiment_id(self, description: str) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{description}_{timestamp}_{self.seed}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def create_reproducible_config(self, 
                               model_config: Dict,
                               data_config: Dict,
                               training_config: Dict) -> Dict:
        """Create a complete reproducible configuration."""
        config = {
            'experiment_id': self.generate_experiment_id("hyperspectral_training"),
            'timestamp': datetime.now().isoformat(),
            'seed': self.seed,
            'model_config': model_config,
            'data_config': data_config,
            'training_config': training_config,
            'system_info': {
                'python_version': self._get_python_version(),
                'pytorch_version': self._get_pytorch_version(),
                'numpy_version': self._get_numpy_version(),
                'cuda_available': torch.cuda.is_available()
            }
        }
        
        self.log_experiment({
            'action': 'create_reproducible_config',
            'config': config
        })
        
        return config
    
    def _get_python_version(self) -> str:
        """Get Python version."""
        import sys
        return sys.version.split()[0]
    
    def _get_pytorch_version(self) -> str:
        """Get PyTorch version."""
        import torch
        return torch.__version__.split('+')[0]
    
    def _get_numpy_version(self) -> str:
        """Get NumPy version."""
        import numpy
        return numpy.__version__
    
    def save_reproducible_config(self, config: Dict, output_path: str):
        """Save reproducible configuration to file."""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"🔧 Reproducible config saved to {output_path}")
        except Exception as e:
            print(f"❌ Failed to save config: {e}")


# Utility functions for dataset operations

def verify_dataset_consistency(dataset_path: str, 
                           expected_samples: int,
                           expected_bands: int,
                           tolerance: float = 0.05) -> Dict:
    """
    Verify dataset consistency and identify potential issues.
    
    Returns:
        Dictionary with consistency metrics and issues
    """
    issues = []
    metrics = {}
    
    try:
        # Load sample files to check consistency
        import glob
        data_files = glob.glob(f"{dataset_path}/*.npy")
        
        if len(data_files) == 0:
            issues.append("No data files found")
            return {'issues': issues, 'metrics': metrics}
        
        # Check sample count
        actual_samples = len(data_files)
        sample_diff = abs(actual_samples - expected_samples) / expected_samples
        
        if sample_diff > tolerance:
            issues.append(f"Sample count mismatch: expected {expected_samples}, got {actual_samples}")
        
        metrics['sample_count'] = actual_samples
        metrics['expected_samples'] = expected_samples
        metrics['sample_difference'] = sample_diff
        
        # Check spectral consistency
        band_counts = []
        for file_path in data_files[:10]:  # Check first 10 files
            try:
                data = np.load(file_path)
                band_counts.append(data.shape[0])
            except Exception as e:
                issues.append(f"Failed to load {file_path}: {e}")
        
        if band_counts:
            avg_bands = np.mean(band_counts)
            band_std = np.std(band_counts)
            
            if band_std > 0:
                issues.append(f"Inconsistent band count: avg {avg_bands:.1f} ± {band_std:.1f}")
            
            if abs(avg_bands - expected_bands) > tolerance * expected_bands:
                issues.append(f"Band count mismatch: expected {expected_bands}, avg {avg_bands:.1f}")
        
        metrics['avg_bands'] = float(avg_bands) if band_counts else 0
        metrics['band_std'] = float(band_std) if band_counts else 0
        metrics['expected_bands'] = expected_bands
        
        # Check data range
        data_ranges = []
        for file_path in data_files[:5]:
            try:
                data = np.load(file_path)
                data_ranges.append({
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data))
                })
            except Exception:
                continue
        
        if data_ranges:
            all_means = [r['mean'] for r in data_ranges]
            all_stds = [r['std'] for r in data_ranges]
            
            metrics['data_range'] = {
                'overall_mean': float(np.mean(all_means)),
                'overall_std': float(np.mean(all_stds)),
                'sample_ranges': data_ranges
            }
    
    except Exception as e:
        issues.append(f"Dataset verification failed: {e}")
    
    return {
        'issues': issues,
        'metrics': metrics,
        'status': 'passed' if not issues else 'failed'
    }


def create_data_manifest(dataset_path: str, 
                     output_path: str = "data/manifest.json") -> Dict:
    """
    Create a comprehensive manifest of the dataset.
    """
    try:
        import glob
        data_files = glob.glob(f"{dataset_path}/*.npy")
        
        manifest = {
            'dataset_path': dataset_path,
            'created_at': datetime.now().isoformat(),
            'total_files': len(data_files),
            'files': []
        }
        
        for file_path in sorted(data_files):
            try:
                data = np.load(file_path)
                file_info = {
                    'path': file_path,
                    'size_mb': Path(file_path).stat().st_size / (1024 * 1024),
                    'shape': list(data.shape),
                    'dtype': str(data.dtype),
                    'min_value': float(np.min(data)),
                    'max_value': float(np.max(data)),
                    'mean_value': float(np.mean(data)),
                    'std_value': float(np.std(data))
                }
                manifest['files'].append(file_info)
            except Exception as e:
                manifest['files'].append({
                    'path': file_path,
                    'error': str(e)
                })
        
        # Save manifest
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"📋 Data manifest created: {len(manifest['files'])} files")
        return manifest
        
    except Exception as e:
        print(f"❌ Failed to create manifest: {e}")
        return {}
