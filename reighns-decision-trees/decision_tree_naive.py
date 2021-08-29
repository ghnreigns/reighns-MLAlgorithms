import os
import sys  # noqa

sys.path.append(os.getcwd())  # noqa
from typing import *
import importlib

Entropy = importlib.import_module("reighns-utils.entropy", package="reighns-utils")


import numpy as np


def partition_entropy(subsets: List[List[Any]]) -> float:
    """Returns the entropy from this partition of data into subsets"""
    total_count = sum(len(subset) for subset in subsets)
    return sum(
        Entropy.calculate_entropy(subset) * len(subset) / total_count
        for subset in subsets
    )


def info_gain(data, labels, attribute, attributes):
    """
    Args:
      data(m, D): The current sub-dataset of the sub-tree
          m: num_rows
          D: num_features
      labels(m): The corresponding labels of current sub-dataset of the sub-tree
          m: num_rows
      attribute: The attribute used to split data
      attributes: The list of current remaining attributes
    Returns:
      info_gain : information gain of the given dataset
    """

    freq = {}
    subsetEntropy = 0.0

    # Get the column index of this attribute
    i = attributes.index(attribute)

    for entry in data:
        if entry[i] in freq:
            freq[entry[i]] += 1.0
        else:
            freq[entry[i]] = 1.0

    ###########################
    #
    # Your Turn (Q6): Write your code here
    # Hint: Split the data based on the value at index i. Find the subsetEntropy of
    # each sub-dataset, then use the formula of Information Gain to calculate subsetEntropy.
    #

    ###########################
    return entropy(labels) - subsetEntropy
