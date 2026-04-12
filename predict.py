"""
predict.py — Run inference on a hyperspectral field scan (.hdr file)
using a trained SoilHSI3DCNN checkpoint.

Usage:
    python predict.py --hdr field_01.hdr --checkpoint runs/best_model.pth
    python predict.py --hdr field_01.hdr --checkpoint runs/best_model.pth --save-map

Outputs:
    • Per-patch soil health class and confidence
    • Per-patch contaminant probabilities (PE, PP, PS/PET, Other)
    • Optional: contamination heatmap saved as a .png
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import yaml
from scipy.ndimage import zoom

from configs.constants import HEALTH_LABELS, CONTAM_LABELS, MODEL_DEFAULTS


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_hdr(hdr_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load an ENVI .hdr hyperspectral cube.
    Returns:
        cube        : float32 array of shape (rows, cols, bands)
        wavelengths : 1-D float array of band centres in nm
    """
    try:
        import spectral.io.envi as envi
    except ImportError:
        sys.exit("❌  spectral not installed. Run: pip install spectral")

    img = envi.open(hdr_path)
    cube = img.load().astype(np.float32)          # (rows, cols, bands)
    metadata = img.metadata

    if "wavelength" in metadata:
        wavelengths = np.array(metadata["wavelength"], dtype=float)
    else:
        print("⚠️  No wavelength metadata — estimating 400–2500 nm range")
        wavelengths = np.linspace(400, 2500, cube.shape[2])

    print(f"✅ Loaded: {hdr_path}")
    print(f"   Spatial  : {cube.shape[0]} × {cube.shape[1]} px")
    print(f"   Bands    : {cube.shape[2]}")
    print(f"   λ range  : {wavelengths[0]:.1f} – {wavelengths[-1]:.1f} nm")
    return cube, wavelengths


def preprocess_cube(cube: np.ndarray, num_bands: int, target_size: tuple) -> torch.Tensor:
    """
    Resize → transpose to (bands, H, W) → normalise → add batch+channel dims.
    Returns tensor of shape (1, 1, num_bands, H, W) ready for the model.
    """
    rows, cols, bands = cube.shape
    tH, tW = target_size

    # Spatial resize
    if (rows, cols) != (tH, tW):
        cube = zoom(cube, (tH / rows, tW / cols, 1), order=1)

    # (H, W, B) → (B, H, W)
    cube = cube.transpose(2, 0, 1)

    # Band resample
    if cube.shape[0] != num_bands:
        cube = zoom(cube, (num_bands / cube.shape[0], 1, 1), order=1)

    # Band-wise z-score (same as dataset.py)
    mean = cube.mean(axis=(1, 2), keepdims=True)
    std  = cube.std(axis=(1, 2), keepdims=True) + 1e-8
    cube = (cube - mean) / std

    # (B, H, W) → (1, 1, B, H, W)
    tensor = torch.from_numpy(cube).unsqueeze(0).unsqueeze(0)
    return tensor


def load_model(checkpoint_path: str, cfg: dict, device: torch.device):
    """Load SoilHSI3DCNN from a .pth checkpoint."""
    try:
        from model import SoilHSI3DCNN
    except ImportError:
        sys.exit("❌  Cannot import model.py. Run predict.py from the repo root.")

    model = SoilHSI3DCNN(
        num_bands=cfg["model"]["num_bands"],
        num_classes=len(HEALTH_LABELS),
        num_contaminants=len(CONTAM_LABELS),
    )

    state = torch.load(checkpoint_path, map_location=device)
    # Support both raw state_dict and wrapped checkpoint dicts
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"✅ Checkpoint loaded: {checkpoint_path}")
    return model


def run_inference(model, tensor: torch.Tensor, device: torch.device) -> dict:
    tensor = tensor.to(device)
    with torch.no_grad():
        health_logits, contam_probs = model(tensor)

    health_probs   = torch.softmax(health_logits, dim=1).cpu().numpy()[0]
    contam_probs   = contam_probs.cpu().numpy()[0]
    health_class   = int(health_probs.argmax())
    health_conf    = float(health_probs.max())

    return {
        "health_class":  health_class,
        "health_label":  HEALTH_LABELS[health_class],
        "health_conf":   health_conf,
        "health_probs":  health_probs,
        "contam_probs":  contam_probs,
    }


def print_results(results: dict) -> None:
    print("\n" + "═" * 48)
    print("  SOIL HEALTH")
    print("═" * 48)
    print(f"  Prediction : {results['health_label']}")
    print(f"  Confidence : {results['health_conf']*100:.1f}%")
    print()
    print("  All health class probabilities:")
    for i, prob in enumerate(results["health_probs"]):
        bar = "█" * int(prob * 20)
        print(f"    [{i}] {HEALTH_LABELS[i]:<25} {prob*100:5.1f}%  {bar}")

    print("\n" + "═" * 48)
    print("  CONTAMINANT DETECTION")
    print("═" * 48)
    for label, prob in zip(CONTAM_LABELS, results["contam_probs"]):
        flag = "⚠️ " if prob > 0.5 else "   "
        bar  = "█" * int(prob * 20)
        print(f"  {flag}{label:<30} {prob*100:5.1f}%  {bar}")
    print("═" * 48 + "\n")


def save_heatmap(results: dict, output_path: str) -> None:
    """Save a simple bar-chart heatmap of contaminant probabilities."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib not installed — skipping heatmap save.")
        return

    probs  = results["contam_probs"]
    colors = ["#e74c3c" if p > 0.5 else "#2ecc71" for p in probs]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(CONTAM_LABELS, probs * 100, color=colors)
    ax.axvline(50, color="gray", linestyle="--", linewidth=0.8, label="50% threshold")
    ax.set_xlabel("Detection probability (%)")
    ax.set_title(f"Contaminant Map — Health: {results['health_label']} "
                 f"({results['health_conf']*100:.1f}%)")
    ax.set_xlim(0, 100)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✅ Heatmap saved: {output_path}")
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SoilHSI3DCNN inference on a .hdr hyperspectral scan"
    )
    parser.add_argument("--hdr",        required=True,  help="Path to .hdr file")
    parser.add_argument("--checkpoint", required=True,  help="Path to .pth model checkpoint")
    parser.add_argument("--config",     default="configs/default.yaml",
                        help="Path to config YAML (default: configs/default.yaml)")
    parser.add_argument("--save-map",   action="store_true",
                        help="Save a contaminant probability heatmap as .png")
    parser.add_argument("--output",     default=None,
                        help="Output path for heatmap (default: <hdr_stem>_map.png)")
    parser.add_argument("--cpu",        action="store_true",
                        help="Force CPU inference even if CUDA is available")
    return parser.parse_args()


def main():
    args = parse_args()

    # Device
    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() \
             else torch.device("cuda")
    print(f"🖥️  Device: {device}")

    # Config
    cfg = load_config(args.config)
    num_bands   = cfg["model"]["num_bands"]
    target_size = tuple(cfg["model"]["target_size"])

    # Load data
    cube, _ = load_hdr(args.hdr)

    # Preprocess
    tensor = preprocess_cube(cube, num_bands, target_size)

    # Load model
    model = load_model(args.checkpoint, cfg, device)

    # Inference
    results = run_inference(model, tensor, device)

    # Print
    print_results(results)

    # Optional heatmap
    if args.save_map:
        stem = Path(args.hdr).stem
        out  = args.output or f"{stem}_map.png"
        save_heatmap(results, out)


if __name__ == "__main__":
    main()
