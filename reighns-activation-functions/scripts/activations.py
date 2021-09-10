import numpy as np


class Sigmoid:
    def __call__(self, z):
        sigmoid = 1 / (1 + np.exp(-z))
        return sigmoid

    def gradient(self, z):
        """
        Compute the gradient of the sigmoid function with respect to the input z.
        This is useful since in the backward pass for say Logistic Regression's Cross-Entropy Loss,
        dl/dz is needed in the chain rule, and dl/dz = dl/dA * dA/dz where A is y_pred is sigmoid(z).
        Consequently, dA/dz makes use of the gradient of the sigmoid function.

                Parameters:
                        z (np.array): 2D numpy array (n_samples, n_features). Input Matrix of size m by n; where m is the number of samples, and n the number of features.

                Returns:
                        dsigmoid_dz (scalar): gradient of sigmoid function at the input z.

        """
        sigmoid = self.__call__(z)  # call sigmoid(z) = 1/ (1+e^{-z})
        # the derivative of sigmoid(z) is sigmoid'(z) = sigmoid(z) * (1-sigmoid(z))
        dsigmoid_dz = sigmoid * (1 - sigmoid)
        return dsigmoid_dz
