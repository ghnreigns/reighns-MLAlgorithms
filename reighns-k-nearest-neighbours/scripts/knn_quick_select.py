import os
import random
import sys  # noqa

sys.path.append(os.getcwd())  # noqa
import importlib
import statistics

DistanceMetrics = importlib.import_module(
    "reighns-distance-metrics.distance", package="reighns-distance-metrics"
)


def partition(arr, lo, high):
    rand = random.randint(lo, high)
    arr[rand], arr[high] = arr[high], arr[rand]
    pivot = lo
    for i in range(lo, high):
        if arr[i][0] < arr[high][0]:
            arr[i], arr[pivot] = arr[pivot], arr[i]
            pivot += 1
    arr[pivot], arr[high] = arr[high], arr[pivot]
    return pivot


def quickselect(arr, lo, hi, k):
    while True:
        pivot = partition(arr, lo, hi)
        if pivot < k:
            lo = pivot + 1
        elif pivot > k:
            hi = pivot - 1
        else:
            return


def KNN_quickselect(X, y, x_test, k):
    # first, calculate the distance from the new point
    # to all other points in the dataset (x_test)
    distances = []  # stores (dist, class)
    for i in range(len(X)):
        d = DistanceMetrics.euclidean_distance(X[i], x_test)
        distances.append((d, y[i]))

    # second, sort and then store the K nearest neighbors
    quickselect(distances, 0, len(distances), k)
    # Get the most popular class
    classes = [c for dist, c in distances[:k]]
    y_pred = statistics.mode(classes)
    return y_pred
