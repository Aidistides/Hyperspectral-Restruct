import numpy as np

def snv(spectra):
    """
    Standard Normal Variate normalization
    """
    mean = np.mean(spectra, axis=1, keepdims=True)
    std = np.std(spectra, axis=1, keepdims=True)
    return (spectra - mean) / std


def msc(spectra):
    """
    Multiplicative Scatter Correction
    """
    ref = np.mean(spectra, axis=0)

    corrected = []
    for spectrum in spectra:
        fit = np.polyfit(ref, spectrum, 1)
        corrected.append((spectrum - fit[1]) / fit[0])

    return np.array(corrected)
