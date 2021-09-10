import importlib
import os
import sys  # noqa
from typing import List

import numpy as np

sys.path.append(os.getcwd())  # noqa

LossFunction = importlib.import_module(
    "reighns-loss-functions.scripts.loss_functions", package="reighns-loss-functions.scripts"
)

utils = importlib.import_module("reighns-utils.scripts.utils", package="reighns-utils.scripts")


class reighnsLinearRegression:
    """
    Linear Regression class generalized to n-features. For description, read the method fit.
    ...
    Attributes
    ----------
    coef_ : float
        the coefficient vector
    intercept_ : float
        the intercept value
    has_intercept : bool
        whether to include intercept or not
    _fitted: bool
        a flag to turn to true once we called fit on the data
    Methods
    -------
    fit(self, X: np.ndarray = None, y_true: np.ndarray = None):
        fits the model and calculates the coef and intercept.
    """

    def __init__(
        self,
        has_intercept: bool = True,
        solver: str = "Closed Form Solution",
        learning_rate: float = 0.1,
        loss_function: LossFunction = LossFunction.l2_loss,
        regularization: int = None,
        num_epochs: int = 1000,
    ):
        """
        Constructs all the necessary attributes for the LinearRegression object.
        Parameters
        ----------
            has_intercept : bool
                whether to include intercept or not

            solver: str
                One of {"Closed Form Solution", "Batch Gradient Descent", "Stochastic Gradient Descent"}

                if Closed Form Solution: closed form solution for finding optimal parameters of beta
                                         recall \vec{beta} = (X'X)^{-1}X'Y ; note scikit-learn uses a slightly different way.
                                         https://stackoverflow.com/questions/66881829/implementation-of-linear-regression-closed-form-solution/66886954#66886954
        """

        super().__init__()

        self.solver = solver
        self.has_intercept = has_intercept
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.regularization = regularization
        self.num_epochs = num_epochs

        self.coef_ = None
        self.intercept_ = None
        self._fitted = False
        self.optimal_betas = None

        self._dft = None
        self._dfe = None
        self._residuals = None
        self.loss_history: List = []  # keep track of loss to plot

    def _add_penalty(self, loss, w: np.ndarray):
        """Apply regularization to the loss."""
        if self.penalty == "l1":
            loss += self.C * np.abs(w[1:]).sum()
        elif self.penalty == "l2":
            loss += (0.5 * self.C) * (w[1:] ** 2).sum()
        return loss

    def _init_weights(self, X: np.ndarray):
        """
        To be included for Gradient Descent
        """
        n_features = X.shape[1]
        # init with all 0s
        initial_weights = np.zeros(
            shape=(1, n_features)
        )  # 1d array is not good, make it 2d array, or a 1 x n matrix
        return initial_weights

    def check_shape(self, X: np.ndarray, y_true: np.ndarray):
        """
        Check the shape of the inputs X & y_true
        if X is 1D array, then it is simple linear regression, reshape to 2D
        [1,2,3] -> [[1],[2],[3]] to fit the data
                Parameters:
                        X (np.ndarray): 2D numpy array (n_samples, n_features). Input Matrix of size m by n; where m is the number of samples, and n the number of features.
                        y_true (np.ndarray): 1D numpy array (n_samples,). Input ground truth, also referred to as y_true of size m by 1.
                Returns:
                        self: Method for chaining
                Examples:
                --------
                        >>> see main
                Explanation:
                -----------
        """

        if X is not None and len(X.shape) == 1:
            X = X.reshape(-1, 1)

        if y_true is not None and len(y_true.shape) == 1:
            y_true = np.reshape(y_true, newshape=(-1, 1))

        return X, y_true

    def degrees_of_freedom(self, X: np.ndarray, y_true: np.ndarray):
        """[summary]

        Args:
            X (np.ndarray): [description]
            y_true (np.ndarray): [description]
        """

        # # degrees of freedom of population dependent variable variance
        # self._dft = self._features.shape[0] - 1
        # # degrees of freedom of population error variance
        # self._dfe = self._features.shape[0] - self._features.shape[1] - 1

    def fit(self, X: np.ndarray = None, y_true: np.ndarray = None):
        """
        Does not return anything. Instead it calculates the optimal beta coefficients for the Linear Regression Model. The default solver will be the closed formed solution
        B = (XtX)^{-1}Xty where we guarantee that this closed solution is unique, provided the invertibility of XtX. This is also called the Ordinary Least Squares Estimate
        where we minimze the Mean Squared Loss function to get the best beta coefficients which gives rise to the least loss function.
                Parameters:
                        X (np.ndarray): 2D numpy array (n_samples, n_features). Input Matrix of size m by n; where m is the number of samples, and n the number of features.
                        y_true (np.ndarray): 1D numpy array (n_samples,). Input ground truth, also referred to as y_true of size m by 1.
                Returns:
                        self (MyLinearRegression): Method for chaining, as you must call .fit first on the LinearRegression class.
                                                   https://stackoverflow.com/questions/36250990/return-self-in-python
                Examples:
                --------
                        >>> see main
                Explanation:
                -----------
        """

        X, y_true = self.check_shape(X, y_true)

        # add a column of ones if there exists an intercept: recall this is needed for intercept beta_0 whereby each sample is y_i = b1x1+b2x2+...+b0(1)
        if self.has_intercept:
            X = np.insert(X, 0, 1, axis=1)  # np.c_[np.ones(n_samples), X]

        n_samples, n_features = X.shape[0], X.shape[1]

        # X = m x n matrix
        assert X.shape == (n_samples, n_features)
        assert y_true.shape == (n_samples, 1)

        if self.solver == "Closed Form Solution":

            XtX = np.transpose(X, axes=None) @ X
            det = np.linalg.det(XtX)
            if det == 0:
                print("Singular Matrix, Recommend to use SVD")

            XtX_inv = np.linalg.inv(XtX)
            Xty = np.transpose(X, axes=None) @ y_true
            self.optimal_betas = XtX_inv @ Xty

        elif self.solver == "Batch Gradient Descent":
            assert self.num_epochs is not None

            # B^T = n x 1 matrix in order for XB^T to work, closely following PyTorch implementation.
            self.optimal_betas = self._init_weights(X)
            assert self.optimal_betas.T.shape == (n_features, 1)

            for epoch in range(self.num_epochs):
                y_pred = np.matmul(X, self.optimal_betas.T)
                assert y_pred.shape == (n_samples, 1)
                assert y_true.shape == (n_samples, 1)

                loss = self.loss_function(y_true, y_pred)

                # Here we knowingly used l2 loss gradient vector
                # where it is represented as $\nabla(\hat{\beta}) = [\nabla(\beta_1), ...., \nabla(\beta_n)]
                GRADIENT_VECTOR = (2 / n_samples) * -(y_true - y_pred).T @ X
                # yet another vectorized operation
                self.optimal_betas -= self.learning_rate * GRADIENT_VECTOR
                if epoch % 100 == 0:
                    print("EPOCH: {} | MSE_LOSS : {}".format(epoch, loss))
                    self.loss_history.append(loss)

        # set attributes from None to the optimal ones

        self.coef_ = self.optimal_betas[0][1:]
        self.intercept_ = self.optimal_betas[0][0]
        self._fitted = True

        return self

    @utils.NotFitted
    def predict(self, X: np.ndarray):
        """
        Predicts the y_true value given an input of X.
                Parameters:
                        X (np.ndarray): 2D numpy array (n_samples, n_features).
                Returns:
                        y_hat: y_pred
                Examples:
                --------
                        >>> see main
                Explanation:
                -----------
        """
        if self.has_intercept:
            # y_pred = self.intercept_ + X @ self.coef_
            X = np.insert(X, 0, 1, axis=1)
            y_pred = np.matmul(X, self.optimal_betas.T)
        else:
            y_pred = np.matmul(X, self.coef_.T)

        return y_pred

    @utils.NotFitted
    def residuals(self, X: np.ndarray, y_true: np.ndarray):
        """[summary]

        Args:
            X (np.ndarray): [description]
            y_true (np.ndarray): [description]
        """
        self._residuals = y_true - self.predict(X)
