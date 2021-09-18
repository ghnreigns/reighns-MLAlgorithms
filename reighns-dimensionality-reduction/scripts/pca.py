from __future__ import print_function, division
import numpy as np


def calculate_covariance_matrix(X, Y=None):
    """Calculate the covariance matrix for the dataset X"""
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples - 1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))

    return np.array(covariance_matrix, dtype=float)


class PCA:
    """A method for doing dimensionality reduction by transforming the feature
    space to a lower dimensionality, removing correlation between features and
    maximizing the variance along each feature axis. This class is also used throughout
    the project to plot data.

    Principal component analysis (PCA) implementation.

    Transforms a dataset of possibly correlated values into n linearly
    uncorrelated components. The components are ordered such that the first
    has the largest possible variance and each following component as the
    largest possible variance given the previous components. This causes
    the early components to contain most of the variability in the dataset.

    Parameters
    ----------
    n_components : int
    solver : str, default 'svd'
        {'svd', 'eigen'}
    """

    def transform(self, X, n_components):
        """Fit the dataset to the number of principal components specified in the
        constructor and return the transformed dataset"""
        covariance_matrix = calculate_covariance_matrix(X)

        # Where (eigenvector[:,0] corresponds to eigenvalue[0])
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sort the eigenvalues and corresponding eigenvectors from largest
        # to smallest eigenvalue and select the first n_components
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:n_components]
        eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :n_components]

        # Project the data onto principal components
        X_transformed = X.dot(eigenvectors)

        return X_transformed


if __name__ == "__main__":

    # X is a 9 x 2 matrix with 9 samples and 2 features x_1 and x_2
    X = np.array(
        [
            [2.5, 2.4],
            [0.5, 0.7],
            [2.2, 2.9],
            [1.9, 2.2],
            [3.1, 3.0],
            [2.3, 2.7],
            [2.0, 6.0],
            [1.0, 1.0],
            [1.5, 1.6],
            [1.1, 0.9],
        ]
    )

    # x_1 and x_2 are first and second column of X respectively

    x_1 = X[:, 0]
    x_2 = X[:, 1]

    # Step 1:
    print(x_1, x_2)
