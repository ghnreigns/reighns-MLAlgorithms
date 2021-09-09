import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

"""
Generates Linear Regression Population Data that follows Assumptions
"""

torch.manual_seed(1992)

device = "cuda"


def generate_dataset_simple(
    bias: float,
    beta: float,
    num_pop: int,
    num_features: int,
    pop_error_mean: float,
    pop_error_std_dev: float,
):
    """Generates True Population data with one bias and one beta

    X ~ U(0, 100) uniform distribution
    e ~ N(0, 1)   normal distribution of mean 0 and std 1

    Args:
        bias ([type]): [description]
        beta ([type]): [description]
        num_pop ([type]): [description]
        num_features ([type]): [description]
        error_mean ([type]): [description]
        error_std_dev ([type]): [description]
    Returns:
        [type]: [description]
    """

    # Generate x as an array of `n` samples which can take a value between 0 and 100

    x = torch.randint(low=0, high=100, size=(num_pop,)).to(device)

    # Generate the random error of n samples, with a random value from a normal distribution, with mean 0 and std 1.
    # Condition: Homoscedasticity within error terms
    # Condition: Normality of the residuals/true error terms

    e = torch.normal(mean=pop_error_mean, std=pop_error_std_dev, size=(num_pop,)).to(
        device
    )  # * std_dev

    # assert torch.mean(e) == 0, torch.std(e) == 1

    # Condition: Linearity between x and y with an small error term e
    y = x * beta + bias + e
    return x, y, e
