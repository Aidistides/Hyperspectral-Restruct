from .smoothing import savgol_smoothing
from .normalization import snv, msc
from .continuum import batch_continuum_removal
from .filtering import select_wavelength_range

class PreprocessingPipeline:
    def __init__(self,
                 smoothing=True,
                 normalization="snv",
                 continuum=False,
                 wavelength_range=None):
        self.smoothing = smoothing
        self.normalization = normalization
        self.continuum = continuum
        self.wavelength_range = wavelength_range

    def transform(self, spectra, wavelengths=None):

        if self.wavelength_range and wavelengths is not None:
            spectra, wavelengths = select_wavelength_range(
                spectra,
                wavelengths,
                self.wavelength_range[0],
                self.wavelength_range[1]
            )

        if self.smoothing:
            spectra = savgol_smoothing(spectra)

        if self.continuum:
            spectra = batch_continuum_removal(spectra)

        if self.normalization == "snv":
            spectra = snv(spectra)
        elif self.normalization == "msc":
            spectra = msc(spectra)

        return spectra, wavelengths
