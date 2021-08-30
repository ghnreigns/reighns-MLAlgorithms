import os
import sys  # noqa

sys.path.append(os.getcwd())  # noqa
from typing import *
import importlib

Entropy = importlib.import_module(
    "reighns-utils.scripts.entropy", package="reighns-utils"
)


def partition_entropy(subsets: List[List[Any]]) -> float:
    """Returns the entropy from this partition of data into subsets"""
    total_count = sum(len(subset) for subset in subsets)

    return sum(
        Entropy.calculate_entropy(subset) * len(subset) / total_count
        for subset in subsets
    )


import numpy as np

from typing import NamedTuple, Optional


class Candidate(NamedTuple):
    level: str
    lang: str
    tweets: bool
    phd: bool
    did_well: Optional[bool] = None  # allow unlabeled data


#  level     lang     tweets  phd  did_well
#  Senior    Python   False   True  True

inputs = [
    Candidate("Senior", "Java", False, False, False),
    Candidate("Senior", "Java", False, True, False),
    Candidate("Mid", "Python", False, False, True),
    Candidate("Junior", "Python", False, False, True),
    Candidate("Junior", "R", True, False, True),
    Candidate("Junior", "R", True, True, False),
    Candidate("Mid", "R", True, True, True),
    Candidate("Senior", "Python", False, False, False),
    Candidate("Senior", "R", True, False, True),
    Candidate("Junior", "Python", True, False, True),
    Candidate("Senior", "Python", True, True, True),
    Candidate("Mid", "Python", False, True, True),
    Candidate("Mid", "Java", True, False, True),
    Candidate("Junior", "Python", False, True, False),
]

from typing import Dict, TypeVar
from collections import defaultdict

T = TypeVar("T")  # generic type for inputs


def partition_by(inputs: List[T], attribute: str) -> Dict[Any, List[T]]:
    """Partition the inputs into lists based on the specified attribute."""
    partitions: Dict[Any, List[T]] = defaultdict(list)
    for input in inputs:
        key = getattr(input, attribute)  # value of the specified attribute
        partitions[key].append(input)  # add input to the correct partition
    return partitions


def partition_entropy_by(
    inputs: List[Any], attribute: str, label_attribute: str
) -> float:
    """Compute the entropy corresponding to the given partition"""
    # partitions consist of our inputs
    partitions = partition_by(inputs, attribute)

    # but partition_entropy needs just the class labels
    labels = [
        [getattr(input, label_attribute) for input in partition]
        for partition in partitions.values()
    ]

    return partition_entropy(labels)


for key in ["level", "lang", "tweets", "phd"]:
    print(key, partition_entropy_by(inputs, key, "did_well"))

assert 0.69 < partition_entropy_by(inputs, "level", "did_well") < 0.70
assert 0.86 < partition_entropy_by(inputs, "lang", "did_well") < 0.87
assert 0.78 < partition_entropy_by(inputs, "tweets", "did_well") < 0.79
assert 0.89 < partition_entropy_by(inputs, "phd", "did_well") < 0.90

senior_inputs = [input for input in inputs if input.level == "Senior"]

assert 0.4 == partition_entropy_by(senior_inputs, "lang", "did_well")
assert 0.0 == partition_entropy_by(senior_inputs, "tweets", "did_well")
assert 0.95 < partition_entropy_by(senior_inputs, "phd", "did_well") < 0.96
