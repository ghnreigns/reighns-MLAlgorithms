"""
Linear_R: y = wTx+b
Logistic_R: y = 1/(1+exp(-(wTx+b)))
Loss_Fn = Cross Entropy Loss - 1/N summation(y_true log y_pred + (1-y_true)log(1-y_pred))
Gradient Vector = (y_true-y_pred)X

"""
import numpy as np


class Sigmoid:
    def __call__(self, z):
        sigmoid_function = 1 / (1 + np.exp(-z))
        return sigmoid_function


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
    cross_entropy_loss = -np.sum(y_true * np.log(y_pred)) / N
    return cross_entropy_loss


class MyLogisticRegression:
    def __init__(self, learning_rate=0.1, num_epochs=1000, has_intercept=True):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.has_intercept = has_intercept
        self.sigmoid = Sigmoid()

        self.coef_ = None
        self.intercept_ = None
        self.optimal_betas = None
        self._fitted = False

    def _init_weights(self, X: np.ndarray):
        n_features = X.shape[1]
        initial_weights = np.zeros(shape=(n_features))
        return initial_weights

    def fit(self, X, y_true):
        if self.has_intercept:
            X = np.insert(X, 0, 1, axis=1)
        n_samples = X.shape[0]
        self.optimal_betas = self._init_weights(X)

        for epoch in range(self.num_epochs):
            z = X @ self.optimal_betas
            y_pred = self.sigmoid(z)
            CROSS_ENTROPY_LOSS = cross_entropy(
                y_true, y_pred
            )  # here it is already divided by N in the code!
            GRADIENT_VECTOR = (
                y_true - y_pred
            ) @ X  # gives shape of 1xn or 1x(n+1) sigmoid's gradient
            GRADIENT_VECTOR /= n_samples  # to average through all samples
            # vectorizing here
            self.optimal_betas += self.learning_rate * GRADIENT_VECTOR

            if epoch % 100 == 0:
                print(CROSS_ENTROPY_LOSS)

        self.coef_ = self.optimal_betas[1:]
        self.intercept_ = self.optimal_betas[0]
        self._fitted = True

        return self

    def predict(self, X):
        if self.has_intercept:
            X = np.insert(X, 0, 1, axis=1)
            Z = X @ self.optimal_betas
            y_logits = self.sigmoid(Z)
        else:
            y_logits = self.sigmoid(X @ self.optimal_betas)

        y_pred = np.where(y_logits < 0.5, 0, 1)

        return y_logits, y_pred


if __name__ == "__main__":
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=1,
        n_classes=2,
        n_clusters_per_class=1,
    )

    logreg = LogisticRegression(fit_intercept=True, random_state=1930)
    logreg.fit(X, y)

    # print 'Logreg intercept:',
    print(logreg.coef_, logreg.intercept_)
    # print 'Logreg predicted probabilities:', logreg.predict_proba(X[0:5,:])

    mylog = MyLogisticRegression(
        learning_rate=0.1, has_intercept=True, num_epochs=3000)
    mylog.fit(X, y)
    print(mylog.optimal_betas)
    logits, preds = mylog.predict(X)
    print(logreg.score(X, y))
    print(sklearn.metrics.accuracy_score(y, preds))
