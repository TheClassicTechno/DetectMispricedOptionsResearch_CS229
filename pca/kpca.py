import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

class KernelPCAModule:
    def __init__(self, kernel='rbf', n_components=5, **kernel_params):
        self.kernel = kernel
        self.n_components = n_components
        self.kernel_params = kernel_params
        self.scaler = StandardScaler()
        self.model = None
        self.X_fit = None

    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.model = KernelPCA(
            n_components=self.n_components,
            kernel=self.kernel,
            fit_inverse_transform=True,
            **self.kernel_params
        )
        Z = self.model.fit_transform(X_scaled)
        self.X_fit = X_scaled
        return Z

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.transform(X_scaled)

    def explained_variance_proxy(self):
        # Use eigenvalue ratios as variance proxy
        eigenvalues = self.model.eigenvalues_
        total_variance = np.sum(eigenvalues)
        return eigenvalues / total_variance