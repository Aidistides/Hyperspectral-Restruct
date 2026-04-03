import numpy as np

def select_wavelength_range(spectra, wavelengths, min_wl, max_wl):
    """
    Select spectral range
    """
    mask = (wavelengths >= min_wl) & (wavelengths <= max_wl)
    return spectra[:, mask], wavelengths[mask]


def remove_noisy_bands(spectra, threshold=0.01):
    """
    Remove low-variance bands (noise)
    """
    variance = np.var(spectra, axis=0)
    mask = variance > threshold
    return spectra[:, mask]
