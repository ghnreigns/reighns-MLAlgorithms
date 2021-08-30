"""
Linear_R: y = wTx+b
Logistic_R: y = 1/(1+exp(-(wTx+b)))
Loss_Fn = Cross Entropy Loss - 1/N summation(y_true log y_pred + (1-y_true)log(1-y_pred))
Gradient Vector = (y_true-y_pred)X

"""
import numpy as np


class Sigmoid:
    def __call__(self, x: np.array):
        sigmoid_fn = 1 / (1 + np.exp(-x))
        return sigmoid_fn


class MyLogisticRegression:
    def __init__(
        self,
        fit_intercept: bool = True,
        num_epochs: int = 10000,
        learning_rate: float = 0.1,
    ):
        super().__init__()

        self.fit_intercept: bool = fit_intercept
        self.optimal_betas = None
        self.coef_ = None
        self.intercept_ = None
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.sigmoid = Sigmoid()

    def init_weights(self, X):
        """
        come here after we check if X has intercept or not as it will affect shape:
        if two weights, then weight vector = [b1, b2] etc where len(weight_vector) = num of features in X
        """
        n_features = X.shape[1]
        self.optimal_betas = np.ones(shape=(n_features))  # init with zeros
        return self.optimal_betas

    def fit(self, X: np.array, y_true: np.array):
        """Take in X and y"""
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)

        n_samples = X.shape[0]
        self.optimal_betas = self.init_weights(X)  # = [0,0,...]

        for epoch in range(self.num_epochs):
            z = X @ self.optimal_betas
            y_pred = self.sigmoid(z)
            grad_vector = (1 / n_samples) * (y_true - y_pred) @ X  # gradients at X
            self.optimal_betas += self.learning_rate * grad_vector

        self.coef_ = self.optimal_betas[1:]
        self.intercept_ = self.optimal_betas[0]
        self._fitted = True
        return self

    def predict(self, X, threshold=0.5):
        # print(self.coef_)
        y_logits = self.sigmoid(X.dot(self.coef_) + self.intercept_)
        # use round for threshold of 0.5, else define your own classification threshold
        y_pred = np.where(y_logits < threshold, 0, 1)
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
    print(logreg.score(X, y))

    mylog = MyLogisticRegression(
        learning_rate=0.1, fit_intercept=True, num_epochs=30000
    )
    mylog.fit(X, y)
    print(mylog.optimal_betas)
    logits, preds = mylog.predict(X)
    print(sklearn.metrics.accuracy_score(y, preds))