from scipy.spatial import distance
import numpy as np


def manhattan_distance(x_1: np.ndarray, x_2: np.ndarray) -> float:
    """
    Manhattan Distance measures the sum of the absolute differences of two vectors' Cartesian coordinates.

    $$
    L1(a, b) = |a_1-b_1| + |a_2-b_2| + \cdots + |a_n-b_n|
    $$

    It's also referred to as the L1 distance or L1 vector norm.

    Generalizes to N-dimensional Euclidean Space:
    Example: Consider 2 points in 3D (x1,y1, z1), (x2, y2, z2) then the manhanttan distance between them is given by
    |x1-x2| + |y1-y2| + |z1-z2|
    Example:
    x_1 = [1,2,3]
    x_2 = [2,3,5]
    l1(x_1,x_2) = 1 + 1 + 2 = 4

    Args:
        x_1 (np.ndarray): a data point in numpy array of dimension D
        x_2 (np.ndarray): a data point in numpy array of dimension D
    Returns:
        _manhattan_distance (float): the Manhattan distance between the two data points
    """
    _manhattan_distance: float = np.sum(np.abs(x_1 - x_2))
    return _manhattan_distance


def euclidean_distance(x_1: np.ndarray, x_2: np.ndarray) -> float:
    """
    Euclidean Distance measures the length of the line segment bewteen two points in the Euclidean space.  This is also referred to as L2 distance or L2 vector norm.

    $$
    L2(a, b) = \sqrt{(a_1-b_1)^2 + (a_2-b_2)^2 + \cdots + (a_n-b_n)^2}
    $$

    Generalizes to N-dimensional Euclidean Space:
    Example: Consider 2 points in 3D (x1,y1, z1), (x2, y2, z2) then the euclidean distance between them is given by
    \sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)
    In code, one can easily use summation(x_true-x_new)^2 because we can reduce above formula to \sum_{i=1}^N vector_1[i] - vector_2[i] ^2

    Args:
        x_1 (np.ndarray): a data point in numpy array of dimension D
        x_2 (np.ndarray): a data point in numpy array of dimension D
    Returns:
        _euclidean_distance (float): the Euclidean distance between the two data points
    """

    _euclidean_distance: float = np.sqrt(np.sum(np.square(x_1 - x_2)))
    return _euclidean_distance


def cosine_similarity(x_1: np.ndarray, x_2: np.ndarray) -> float:
    assert len(x_1.shape) == len(x_2.shape) in set([1, 2])
    # assert
    origin: np.ndarray = np.zeros(shape=(x_1.shape))
    _cosine_similarity: float = np.dot(x_1, x_2) / (
        euclidean_distance(x_1, origin) * euclidean_distance(x_2, origin)
    )
    return _cosine_similarity


def cosine_distance(x_1: np.ndarray, x_2: np.ndarray) -> float:
    return 1 - cosine_similarity(x_1, x_2)
