import numpy as np
from scipy.signal import savgol_filter

def savgol_smoothing(spectra, window_length=11, polyorder=2):
    """
    Apply Savitzky-Golay smoothing.

    Parameters:
        spectra (np.ndarray): shape (n_samples, n_bands)
    """
    return savgol_filter(spectra, window_length, polyorder, axis=1)
