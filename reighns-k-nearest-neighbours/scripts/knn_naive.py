import os
import sys  # noqa

sys.path.append(os.getcwd())  # noqa
import importlib
import statistics

DistanceMetrics = importlib.import_module(
    "reighns-distance-metrics.distance", package="reighns-distance-metrics"
)


def KNN_example(X, y, x_test, k):
    """[summary]

    Args:
        X ([type]): [description]
        y ([type]): [description]
        x_test ([type]): [description]
        k ([type]): [description]

    Returns:
        [type]: [description]
    """

    # Example input

    # X = [[1,2], [1000,2000], [2,3], [1000,4000]]
    # y = [1,2,1,2]
    # x_test = [3, 4]
    # k = 2

    # first, calculate the distance from the new point
    # to all other points in the dataset (x_test)
    distances = []  # stores (dist, class)
    for i, x_i in enumerate(X):
        d = DistanceMetrics.euclidean_distance(x_i, x_test)
        distances.append((d, y[i]))

    # second, sort and then store the k nearest neighbors
    neighbors = sorted(distances)[:k]
    # Get the most popular class
    classes = [target for dist, target in neighbors]
    y_pred = statistics.mode(classes)
    return y_pred
