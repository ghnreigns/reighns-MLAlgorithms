from math import log2
from typing import *

import numpy as np

# TODO: UNIT TESTING


def class_probabilities(labels: List[any]) -> List[float]:
    """Calculate frequency of each class.

    From DSFS book, it mentions that we do not actually care about which label is associated with which probability. Thus it is okay to use a dictionary which does not preserve order.

    Args:
        labels (List[any]): [description]

    Returns:
        label_probs (List[float]): [description]

    Example:
        labels = ['dog', 'dog', 'cat', 'cat', 'dog'] = [0, 0, 1, 1, 0]
        assert class_probabilities(labels) = [2/5, 3/5] or class_probabilities(labels) = [3/5, 2/5]
    """

    num_samples = len(labels)

    label_count: Dict = {}
    label_probs: List = []

    for label in labels:
        if label not in label_count:
            label_count[label] = 1
        else:
            label_count[label] += 1

    for label, count in label_count.items():

        label_probs.append(count / num_samples)

    return label_probs


def _entropy(
    class_probabilities: List[float], epsilon: float = 1e-15, log_base: int = 2
) -> float:
    """[summary]

    Args:
        class_probabilities (List[float]): Frequency probability of class occurences.

    Returns:
        entropy (float): [description]

    Example:
        # maximum chaos -> entropy = 1
        class_probabilities = [1/2, 1/2]
        assert _entropy(class_probabilities) == 1

        # minimum chaos -> entropy = 0
        class_probabilities = [1, 0] # or [0, 1]
        assert _entropy(class_probabilities) == 0

        class_probabilities = [2/5, 3/5]
        assert _entropy(class_probabilities) == 0.9709505944546686
    """

    assert np.sum(class_probabilities) == 1

    entropy = 0

    for _y in class_probabilities:
        if _y == 0:
            _y = epsilon

        entropy += _y * log2(_y)

    entropy = -1 * entropy

    return entropy


def calculate_entropy(labels: List[any]) -> float:
    """[summary]

    Args:
        labels (List[any]): [description]

    Returns:
        float: [description]
    """

    return _entropy(class_probabilities(labels))
