import matplotlib.pyplot as plt

def plot_spectrum(wavelengths, spectra, sample_id):
    """
    Plot a single spectrum
    """
    plt.figure()
    plt.plot(wavelengths, spectra[sample_id])
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title(f"Spectrum - Sample {sample_id}")
    plt.show()
