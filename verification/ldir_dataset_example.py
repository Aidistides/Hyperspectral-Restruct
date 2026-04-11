"""
Synthetic LDIR Dataset Example for Hyperspectral-Restruct Verification
Based on Jia et al. (2022) - "Automated identification and quantification of invisible microplastics in agricultural soils"

Key numbers from the paper:
- Total MP abundance: ~1.57 × 10^5 to 3.20 × 10^5 particles/kg soil (cotton fields with different years of film mulching)
- Overall study: 47,453 particles observed by LDIR → 34,124 identified as MPs (recognition rate ~71.9%)
- Dominant polymers: PE, PP, PVC, PA (polyamide), PTFE also noted
- Size range: Mostly 10–500 μm (96.5–99.9% of MPs)
- Hit/Match Score filter: Typically ≥65%
- Shapes: Mostly films (~88%), also fibers and pellets
- 26 polymer types detected in total

This module generates realistic synthetic LDIR-style CSV data that you can use to test the LDIRVerifier class.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

def generate_synthetic_ldir_data(
    num_samples: int = 4,
    base_particles_per_sample: int = 8000,  # Scaled down from paper's 47k total for practicality
    output_dir: str = "verification/data",
    seed: int = 42
) -> List[str]:
    """
    Generate synthetic LDIR CSV files mimicking Jia et al. (2022) results.
    
    Returns list of generated file paths.
    """
    np.random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Dominant polymers with realistic proportions (PE most common, followed by PP, PVC, PA)
    polymers = ["PE", "PP", "PVC", "PA", "PTFE", "PET", "PS", "Other"]
    polymer_probs = [0.35, 0.25, 0.15, 0.12, 0.05, 0.04, 0.03, 0.01]  # sums to 1.0
    
    generated_files = []
    
    for sample_idx in range(num_samples):
        # Vary abundance based on film mulching years (higher with longer use)
        multiplier = [1.0, 0.8, 0.9, 1.6][sample_idx]  # roughly matches paper's 1.57–3.20 ×10^5 trend
        num_particles = int(base_particles_per_sample * multiplier)
        
        # Generate particle data
        data = {
            "Particle_ID": range(1, num_particles + 1),
            "Polymer": np.random.choice(polymers, size=num_particles, p=polymer_probs),
            "Diameter_um": np.random.lognormal(mean=4.0, sigma=0.8, size=num_particles).clip(10, 500).astype(int),  # realistic 10-500 μm
            "Area_um2": np.random.lognormal(mean=8.0, sigma=1.2, size=num_particles).astype(int),
            "Match_Score": np.random.uniform(65, 95, size=num_particles).round(1),  # ≥65 as per paper
            "Shape": np.random.choice(["Film", "Fragment", "Fiber", "Pellet"], size=num_particles, p=[0.70, 0.20, 0.07, 0.03]),
            "X_pos": np.random.uniform(0, 1000, size=num_particles).round(1),
            "Y_pos": np.random.uniform(0, 1000, size=num_particles).round(1),
            "Sample_Weight_kg": np.full(num_particles, 0.001),  # assume ~1g subsample for scaling
            "Sample_ID": f"cotton_mulch_{5 + sample_idx*5}years",  # e.g., 5,10,15,20+ years
        }
        
        df = pd.DataFrame(data)
        
        # Add a few "undefined" particles (low match score) to simulate real data (~<10% as per paper)
        num_undefined = int(num_particles * 0.08)
        if num_undefined > 0:
            undef_idx = np.random.choice(df.index, num_undefined, replace=False)
            df.loc[undef_idx, "Match_Score"] = np.random.uniform(30, 64, num_undefined).round(1)
            df.loc[undef_idx, "Polymer"] = "Undefined"
        
        # Save CSV
        filename = f"ldir_synthetic_sample_{sample_idx+1}_{data['Sample_ID'][0]}.csv"
        filepath = output_path / filename
        df.to_csv(filepath, index=False)
        
        generated_files.append(str(filepath))
        
        print(f"✅ Generated {num_particles:,} particles → {filepath.name}")
        print(f"   Dominant polymers: {df['Polymer'].value_counts().head(4).to_dict()}")
    
    print(f"\nSynthetic LDIR dataset created in: {output_path}")
    print("Use these files with LDIRVerifier.load_ldir_data()")
    return generated_files


def create_example_usage_script():
    """Optional: Creates a small ready-to-run example script."""
    script = """# Example: How to use the synthetic data with LDIRVerifier

from verification.ldir_verification import LDIRVerifier
from verification.ldir_dataset_example import generate_synthetic_ldir_data

# 1. Generate synthetic data (run once)
generate_synthetic_ldir_data()

# 2. Load and verify one sample
verifier = LDIRVerifier()
ldir_df = verifier.load_ldir_data("verification/data/ldir_synthetic_sample_1_cotton_mulch_5years.csv")
ldir_stats = verifier.aggregate_ldir_to_sample(ldir_df)

print("\\nSample LDIR Statistics:")
print(f"Total particles: {ldir_stats['total_particles']}")
print(f"Polymer distribution: {ldir_stats['polymer_fraction']}")

# In real workflow: compare with your HSI 3D-CNN prediction map here
"""
    script_path = Path("verification") / "example_ldir_verification_usage.py"
    script_path.write_text(script)
    print(f"✅ Example usage script created: {script_path}")


if __name__ == "__main__":
    generate_synthetic_ldir_data()
    create_example_usage_script()
