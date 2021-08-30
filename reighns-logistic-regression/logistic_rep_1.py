import numpy as np


class Sigmoid:
    def __call__(self, z: np.array):
        # define z = wTx+b
        sigmoid_eqn = 1 / (1 + np.exp(-z))
        return sigmoid_eqn


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
    N = y_pred.shape[0]
    ce = -np.sum(y_true * np.log(y_pred + 1e-9)) / N
    return ce


class MyLogisticRegression:
    def __init__(self, learning_rate, has_intercept, num_epochs):
        super().__init__()
        self.coef_ = None
        self.intercept_ = None
        self.beta_vector = None
        self.has_intercept = has_intercept
        self._fitted = False
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.sigmoid = Sigmoid()

    def _initialize_weights(self, X):
        # no matter what initi, we make the weight vector into the same shape for generalization.
        # beta_vector = [b0, b1, b2,...]
        n_samples, n_features = X.shape[0], X.shape[1]
        self.beta_vector = np.zeros(shape=(n_features))
        return self.beta_vector

    def fit(self, X: np.array, y_true: np.array):
        # takes in a 2d array and 1d array y_true, we use gradient descent to find convergence, and during gradient descent, we update the weights accordingly
        # remember y = wTx + b is linear regression while for logistic regression it is y = 1/1+e^{-z} where z = wTx+b; it is worth noting that
        # this y is called logits, the log of the odds, which can be understood as p(X|y), this  is a well calibrated model with natural probabilities as output
        n_samples, n_features = X.shape[0], X.shape[1]
        if self.has_intercept:
            # as usual, make X have a extra column of 1s if there is bias term
            X = np.insert(arr=X, obj=0, values=1, axis=1)
        # note here the X we pass in we will know if we have bias weight or not
        self.beta_vector = self._initialize_weights(X)
        # print(X, self.beta_vector)
        for epoch in range(self.num_epochs):
            # print(self.beta_vector.shape, print(X.shape))

            z = X @ self.beta_vector
            y_pred = self.sigmoid(
                z=z
            )  # since X is in matrix format with column of ones in first column, it suffices to just do z = wTX and not z= wTX +b
            self.beta_vector += (
                self.learning_rate * (1 / n_samples) * (y_true - y_pred) @ X
            )
            # print(y_true, y_pred)
            if epoch % 100 == 0:
                cross_entropy_loss = -(1.0 / n_samples) * (
                    np.dot(np.log(y_pred), y_true.T)
                    + np.dot(np.log(1 - y_pred), (1 - y_true).T)
                )
                # cross_entropy_loss = cross_entropy(y_true, y_pred)
                print("Cross Entropy Loss: {}".format(cross_entropy_loss))

        self.coef_ = self.beta_vector[1:]
        self.intercept_ = self.beta_vector[0]
        self._fitted = True
        return self

    def predict(self, X):
        # print(self.coef_)
        y_logits = self.sigmoid(X.dot(self.coef_) + self.intercept_)
        # use round for threshold of 0.5, else define your own classification threshold
        y_pred = np.round(y_logits).astype(int)
        return y_logits, y_pred