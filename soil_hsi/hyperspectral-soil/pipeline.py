import numpy as np
from .indices import soil_index_example
from .derivatives import first_derivative, second_derivative
from .dimensionality import PCAReducer

class FeatureEngineeringPipeline:
    def __init__(self,
                 use_indices=True,
                 use_derivatives=False,
                 derivative_order=1,
                 use_pca=False,
                 n_components=10):

        self.use_indices = use_indices
        self.use_derivatives = use_derivatives
        self.derivative_order = derivative_order
        self.use_pca = use_pca

        if use_pca:
            self.pca = PCAReducer(n_components=n_components)

    def transform(self, spectra, wavelengths):
        features = [spectra]

        # Spectral indices
        if self.use_indices:
            idx = soil_index_example(spectra, wavelengths)
            features.append(idx.reshape(-1, 1))

        # Derivatives
        if self.use_derivatives:
            if self.derivative_order == 1:
                deriv = first_derivative(spectra)
            else:
                deriv = second_derivative(spectra)

            features.append(deriv)

        X = np.concatenate(features, axis=1)

        # PCA
        if self.use_pca:
            X = self.pca.fit_transform(X)

        return X
