import os
import sys  # noqa

sys.path.append(os.getcwd())  # noqa

import importlib

DistanceMetrics = importlib.import_module(
    "reighns-distance-metrics.scripts.distance", package="reighns-distance-metrics.scripts"
)

reighnsKNN = importlib.import_module(
    "reighns-k-nearest-neighbours.scripts.knn", package="reighns-distance-metrics.scripts"
)


import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.7, random_state=42
)

sklearn_knn = KNeighborsClassifier(n_neighbors=3)
sklearn_predictions = sklearn_knn.fit(X_train, y_train).score(X_test, y_test)

HN_KNN_CLASSIFICATION = reighnsKNN.reighnsKNN(
    k=3, distance_metric=DistanceMetrics.euclidean_distance, mode="classification"
)
HN_CLASSIFICATION_PREDICTIONS = HN_KNN_CLASSIFICATION.predict(X_train, y_train, X_test)

print(HN_CLASSIFICATION_PREDICTIONS)
print("\nSKLEARN Accuracy score : %.3f" % (sklearn_predictions * 100))
print("\nHN Accuracy score : %.3f" % (accuracy_score(y_test, HN_CLASSIFICATION_PREDICTIONS) * 100))
print()

# print(sklearn_knn.fit(X_train, y_train).predict_proba(X_test))
# print("Recall score : %f" % (sklearn.metrics.recall_score(y_val, preds) * 100))
# print("ROC score : %f\n" % (sklearn.metrics.roc_auc_score(y_val, preds) * 100))
# print(sklearn.metrics.confusion_matrix(y_val, preds))

X = np.array([[0], [1], [2], [3]])

y = np.array([0, 0, 1, 1])
from sklearn.neighbors import KNeighborsRegressor

neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X, y)

print(neigh.predict([[1.5]]))
HN_KNN_REGRESSION = reighnsKNN(
    k=2, distance_metric=DistanceMetrics.euclidean_distance, mode="regression"
)
HN_REGRESSION_PREDICTIONS = HN_KNN_REGRESSION.predict(X, y, np.array([[1.5]]))

print(HN_REGRESSION_PREDICTIONS)
