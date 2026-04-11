# feature_selection.py
"""
N-specific band selection for Hyperspectral-Restruct (SPA & MC-UVE).
Prioritizes SWIR for soil nitrogen prediction.
Integrates directly with dataset.py and model.py.
"""

import numpy as np
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
import argparse
import os

# SWIR-prioritized starting bands (nm) – matches NITROGEN_PREDICTIVE_BANDS.md
SWIR_PRIORITY = np.array([1470, 1478, 1650, 1697, 2050, 2099, 2104, 2170, 2296, 2410])

def load_hsi_data(data_path):
    # Reuse your existing dataset.py loader
    from dataset import HyperspectralDataset
    ds = HyperspectralDataset(data_path, target='nitrogen', transform=None)
    X = ds.X.reshape(len(ds), -1)  # (samples, bands)
    y = ds.y
    wavelengths = ds.wavelengths
    return X, y, wavelengths

def spa_selection(X, y, n_features=15):
    """Successive Projections Algorithm (simplified)"""
    pls = PLSRegression(n_components=10)
    selected = []
    remaining = list(range(X.shape[1]))
    while len(selected) < n_features:
        scores = []
        for i in remaining:
            temp = np.column_stack([X[:, selected], X[:, i]]) if selected else X[:, [i]]
            score = np.mean(cross_val_score(pls, temp, y, cv=5, scoring='r2'))
            scores.append(score)
        best = remaining[np.argmax(scores)]
        selected.append(best)
        remaining.remove(best)
    return np.array(selected)

def mc_uve_selection(X, y, n_features=15, n_mc=100):
    """Monte Carlo Uninformative Variable Elimination"""
    pls = PLSRegression(n_components=10)
    importance = np.zeros(X.shape[1])
    for _ in range(n_mc):
        idx = np.random.choice(len(X), len(X)//2, replace=False)
        pls.fit(X[idx], y[idx])
        coef = np.abs(pls.coef_.ravel())
        importance += coef
    importance /= n_mc
    return np.argsort(importance)[-n_features:]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to HSI dataset folder")
    parser.add_argument("--target", default="nitrogen", help="Target variable")
    parser.add_argument("--method", choices=["spa", "mc_uve"], default="spa")
    parser.add_argument("--n_bands", type=int, default=15)
    args = parser.parse_args()

    X, y, wavelengths = load_hsi_data(args.data)
    print(f"Loaded {X.shape[0]} samples × {X.shape[1]} bands")

    if args.method == "spa":
        selected_idx = spa_selection(X, y, args.n_bands)
    else:
        selected_idx = mc_uve_selection(X, y, args.n_bands)

    selected_nm = wavelengths[selected_idx]
    print("\n=== SELECTED BANDS FOR NITROGEN ===")
    print(selected_nm.tolist())
    print(f"SWIR priority match: {np.isin(selected_nm, SWIR_PRIORITY).mean():.1%}")

    # Save config for easy import into train.py / dataset.py
    np.save("configs/nitrogen_selected_bands.npy", selected_nm)
    print("Saved to configs/nitrogen_selected_bands.npy")
