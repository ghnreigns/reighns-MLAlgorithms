from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split


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


def cross_entropy(y_true, y_pred, epsilon=1e-12):
    """
        https://stackoverflow.com/questions/47377222/what-is-the-problem-with-my-implementation-of-the-cross-entropy-function
        Computes cross entropy between targets (encoded as one-hot vectors)
        and y_pred.
        Input: y_pred (N, k) ndarray
               y_true (N, k) ndarray
        Returns: scalar
        predictions = np.array([[0.25,0.25,0.25,0.25],
                            [0.01,0.01,0.01,0.96]])
    targets = np.array([[0,0,0,1],
                       [0,0,0,1]])
                       ans = 0.71355817782  #Correct answer
    x = cross_entropy(predictions, targets)
    print(np.isclose(x,ans))
    """
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    # take note that y_pred is of shape 1 x n_samples as stated in our framework
    n_samples = y_pred.shape[1]

    # cross entropy function
    cross_entropy_function = (
        y_true * np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
    # cross entropy function here is same shape as y_true and y_pred since we are
    # just performing element wise operations on both of them.
    assert cross_entropy_function.shape == (1, n_samples)

    # we sum up all the loss for each individual sample
    total_cross_entropy_loss = -np.sum(cross_entropy_function, axis=1)
    assert total_cross_entropy_loss.shape == (1,)

    # we then average out the total loss across m samples, but we squeeze it to
    # make it a scalar; squeeze along axis = None since there is no column axix
    average_cross_entropy_loss = np.squeeze(
        total_cross_entropy_loss / n_samples, axis=None)

    # cross_entropy_loss = -np.sum(y_true * np.log(y_pred)) / n_samples
    # print(np.isclose(average_cross_entropy_loss, cross_entropy_loss))
    return average_cross_entropy_loss


class MyLogisticRegression:
    def __init__(
        self,
        has_intercept: bool = True,
        learning_rate: float = 1e-3,
        solver: str = "Gradient Descent",
        num_epochs: int = 1000,
    ):
        super().__init__()

        self.solver: str = solver
        self.num_epochs: int = num_epochs
        self.has_intercept: bool = has_intercept
        self.learning_rate: float = learning_rate

        self.coef_: Optional[np.ndarray[float]] = None
        self.intercept_: Optional[float] = None
        self.optimal_betas: Optional[np.ndarray[float]] = None
        self._fitted: bool = False

        self.sigmoid = Sigmoid()

    def _initiatilize_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize the weight and bias vector.

                Parameters:
                        X (np.array): 2D numpy array (n_samples, n_features). Input Matrix of size m by n; where m is the number of samples, and n the number of features.

                Returns:
                        uniform_weights (np.array): using He.Kaimin initialization

        """
        n_features = np.shape(X)[1]
        limit = 1 / np.sqrt(n_features)
        uniform_weights = np.random.uniform(-limit,
                                            limit, size=(n_features, 1))
        return uniform_weights

    def _check_shape(self, X: np.ndarray, y_true: np.ndarray):
        """
        Check the shape of the inputs X & y_true; In particular, y_true must be reshaped to a row vector.

        1.  If X is 1D array, then it is simple logistic regression with 1 feature/variable,
            we need to make sure that X is reshaped to a 2D array/matrix.
            [1,2,3] (3,) -> [[1],[2],[3]] (3, 1) to fit the data.

        2.  If y_true is 1D array, which may usually be the case, we need to reshape it to 2D array.
            y_true = [1,0,1] (3,) -> y_true = [[1],[0],[1]] (3,1)


                Parameters:
                        X (np.ndarray): 2D numpy array (n_samples, n_features). Input Matrix of size m by n; where m is the number of samples, and n the number of features.

                        y_true (np.ndarray): 1D numpy array (n_samples,). Input ground truth, also referred to as y_true of size m by 1.

                Returns:
                        X (np.ndarray):
                        y_true (np.ndarray):
                Examples:
                --------
                        >>> see main

                Explanation:
                -----------

        """
        if X is not None and len(X.shape) == 1:
            X = np.reshape(X, newshape=(-1, 1))

        if y_true is not None and len(y_true.shape) == 1:
            y_true = np.reshape(y_true, newshape=(1, -1))

        return X, y_true

    def fit(self, X: np.ndarray, y_true: np.ndarray):
        """
        Does not return anything. Instead it calculates the optimal beta coefficients for the Logistic Regression Model.
        The default solver will be Batch Gradient Descent where we optimize the weights by minimizing the cross-entropy loss function.

                Parameters:
                        X (np.array): 2D numpy array (n_samples, n_features). Input Matrix of size m by n; where m is the number of samples, and n the number of features.

                        y (np.array): 1D numpy array (n_samples,). Input ground truth, also referred to as y_true of size m by 1.

                Returns:
                        self (MyLogisticRegression): Method for chaining

                Examples:
                --------
                        >>> see main

                Explanation:
                -----------

        """

        X, y_true = self._check_shape(X, y_true)

        if self.has_intercept:
            # Equivalent: X = np.c_[np.ones(n_samples), X]
            X = np.insert(X, 0, 1, axis=1)

        n_samples, n_features = X.shape
        # y_true must be a row vector with shape 1 x n_samples
        assert y_true.shape == (1, n_samples)

        self.optimal_betas = self._initiatilize_weights(X)
        # weight vector must be a column vector of shape (n_features, 1)
        assert self.optimal_betas.shape == (n_features, 1)

        for epoch in range(self.num_epochs):
            z = np.matmul(X, self.optimal_betas).T
            # z must be a row vector with shape 1 x n_samples
            assert z.shape == (1, n_samples)

            y_pred = self.sigmoid(z)
            # y_pred must be a row vector with shape 1 x n_samples
            assert y_pred.shape == (1, n_samples)

            GRADIENT_VECTOR = -np.matmul((y_true - y_pred
                                          ), X).T
            # gradient vector must be a column vector of (n_features, 1)
            assert GRADIENT_VECTOR.shape == (n_features, 1)
            # we need to divide gradient vector by number of samples.
            # this is because each element inside the gradient vector is an accumulation/sum of across all samples.
            AVG_GRADIENT_VECTOR = (1 / n_samples) * GRADIENT_VECTOR

            if self.solver == "Gradient Descent":
                self.optimal_betas -= self.learning_rate * AVG_GRADIENT_VECTOR
                cross_entropy_loss = cross_entropy(y_true, y_pred)

                if epoch % 1000 == 0:
                    print("epoch: {} | loss: {}".format(
                        epoch, cross_entropy_loss))

            self.coef_ = self.optimal_betas[1:]
            self.intercept_ = self.optimal_betas[0]
            self._fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using Logistic Regression - can only be called after fitting.
        Prediction formula will be using the sigmoid function for binary.

                Parameters:
                        X (np.array): 2D numpy array (n_samples, n_features). Input Matrix of size m by n; where m is the number of samples, and n the number of features.

                Returns:
                        y_logits (np.array): raw logits (probabilities).
                        y_pred (np.array): predictions in 0 and 1 for binary.
                -----------

        """
        if self.has_intercept:
            z = np.matmul(X, self.coef_) + self.intercept_  # z is a logit
            y_probs = self.sigmoid(z)
        else:
            z = np.matmul(X, self.coef_)
            y_probs = self.sigmoid(z)

        y_pred = np.where(y_probs < 0.5, 0, 1)

        return y_probs, y_pred


if __name__ == "__main__":

    """
    ================================
    Breast Cancer Classification Exercise
    ================================

    A tutorial exercise regarding the use of classification techniques on
    the Breast Cancer dataset.
    """
    np.random.seed(1930)
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=1930
    )
    print(y_train.shape)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)

    logreg = LogisticRegression(
        fit_intercept=True, random_state=1930, solver="sag", max_iter=1000
    )
    logreg.fit(X_train, y_train)
    print(logreg.coef_, logreg.intercept_)
    print("SKLEARN Validation Accuracy: {}".format(logreg.score(X_val, y_val)))

    mylog = MyLogisticRegression(
        learning_rate=0.1, has_intercept=True, num_epochs=5000)
    mylog.fit(X_train, y_train)

    print(mylog.coef_)
    logits, preds = mylog.predict(X_val)

    print(
        "\nAccuracy score : %f" % (
            sklearn.metrics.accuracy_score(y_val, preds) * 100)
    )
    print("Recall score : %f" %
          (sklearn.metrics.recall_score(y_val, preds) * 100))
    print("ROC score : %f\n" %
          (sklearn.metrics.roc_auc_score(y_val, preds) * 100))
    print(sklearn.metrics.confusion_matrix(y_val, preds))
    """
    ================================
    Digits Classification Exercise
    ================================

    A tutorial exercise regarding the use of classification techniques on
    the Digits dataset.

    This exercise is used in the :ref:`clf_tut` part of the
    :ref:`supervised_learning_tut` section of the
    :ref:`stat_learn_tut_index`.
    """
    # print(__doc__)

    # from sklearn import datasets, neighbors, linear_model

    # """
    # This below is a multiclass logistic regression using softmax. My code
    # not ready yet.
    # """

    # X_digits, y_digits = datasets.load_digits(return_X_y=True)

    # X_digits = X_digits / X_digits.max()

    # n_samples = len(X_digits)

    # X_train = X_digits[: int(0.9 * n_samples)]
    # y_train = y_digits[: int(0.9 * n_samples)]
    # X_test = X_digits[int(0.9 * n_samples) :]
    # y_test = y_digits[int(0.9 * n_samples) :]

    # knn = neighbors.KNeighborsClassifier()
    # logistic = linear_model.LogisticRegression(max_iter=1000)

    # print("KNN score: %f" % knn.fit(X_train, y_train).score(X_test, y_test))
    # print(
    #     "LogisticRegression score: %f"
    #     % logistic.fit(X_train, y_train).score(X_test, y_test)
    # )

    # hn_logreg = MyLogisticRegression(
    #     has_intercept=True,
    #     learning_rate=1e-5,
    #     solver="Gradient Descent",
    #     num_epochs=1000,
    # )
    # hn_logreg.fit(X_train, y_train)
    # ylogits, ypreds = hn_logreg.predict(X_test)
    # print(y_test)
    # print(hn_logreg.coef_)
    # print(ylogits)
    # print(sklearn.metrics.accuracy_score(y_test, ypreds))
