# Trains using PyTorch and logs training metrics and weights in TensorFlow event format to the MLflow run's artifact directory.
# This stores the TensorFlow events in MLflow for later access using TensorBoard.
#
# Code based on https://github.com/mlflow/mlflow/blob/master/example/tutorial/pytorch_tensorboard.py.

from __future__ import print_function

import os
import sys
import tempfile
from argparse import Namespace
from collections import namedtuple
from pathlib import Path
from typing import *

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import typer
from chardet.universaldetector import UniversalDetector
from sklearn import datasets, linear_model
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())  # noqa

import importlib

reighnsLinearRegression = importlib.import_module(
    "reighns-linear-regression.scripts.linear_regression",
    package="reighns-linear-regression.scripts",
)
LossFunction = importlib.import_module(
    "reighns-loss-functions.scripts.loss_functions", package="reighns-loss-functions.scripts"
)


"""
Notice that the intercept and coefficient values are not exactly the same when comparing sklearn's method and mine.
This is because we are using slightly different ways to solve the question.
Normal Equation HN: solving the normal equations by directly inverting the X.T @ X matrix.
Normal Equation SKLEARN: On the other hand, scikit-learn uses scipy.linalg.lstsq under the hood, which uses for example an SVD-based approach.
                            That is, the mechanism there does not invert the matrix and is therefore different than yours.
                            Note that there are many ways to solve the linear least squares problem.
"""


def regression_test(solver: str = "Closed Form Solution", num_epochs: int = None):
    np.random.seed(1930)
    X, y_true = make_regression(
        n_samples=10000, n_features=10, random_state=1930, coef=False
    )  # returns 2-d array of 1000 by 10
    x_train, x_val, y_train, y_val = train_test_split(X, y_true, test_size=0.3, random_state=1930)

    lr_SKLEARN = LinearRegression(fit_intercept=True, normalize=False).fit(x_train, y_train)
    lr_HONGNAN = reighnsLinearRegression.reighnsLinearRegression(
        solver=solver, has_intercept=True, num_epochs=num_epochs
    ).fit(x_train, y_train)

    """
    Debugged, the intercept is the one with major difference why? Answer: https://stackoverflow.com/questions/66881829/implementation-of-linear-regression-closed-form-solution/66886954?noredirect=1#comment118259946_66886954
    """
    print(lr_SKLEARN.intercept_)
    print(lr_HONGNAN.intercept_)

    pred_HONGNAN = lr_HONGNAN.predict(x_val)
    pred_SKLEARN = lr_SKLEARN.predict(x_val)
    print("First Value HN", pred_HONGNAN[0])
    print("First Value SKLEARN", pred_SKLEARN[0])
    print("HN MSE", mean_squared_error(y_val, pred_HONGNAN))
    print("SKLEARN MSE", mean_squared_error(y_val, pred_SKLEARN))


def regression_diabetes():
    # Load the diabetes dataset
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

    # Use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()
    lr_HONGNAN = reighnsLinearRegression.reighnsLinearRegression(
        has_intercept=True, solver="Batch Gradient Descent", num_epochs=6666
    ).fit(diabetes_X_train, diabetes_y_train)

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)
    pred_HONGNAN = lr_HONGNAN.predict(diabetes_X_test)

    # The coefficients
    print("SKLEARN Coefficients:", regr.coef_)
    print("HONGNAN Coefficients:", lr_HONGNAN.coef_)
    # The mean squared error
    print("HN MSE", mean_squared_error(diabetes_y_test, pred_HONGNAN))
    print("SKLEARN MSE", mean_squared_error(diabetes_y_test, diabetes_y_pred))

    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

    # Plot outputs
    plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
    plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()


def regression_conditional_mean(
    bias=11.1,
    beta=9.33,
    num_pop=100000000,
    num_features=1,
    pop_error_mean=0,
    pop_error_std_dev=1,
    conditional_x_value=10,
    num_sample=1000000,
):
    """Refer to the Conditional Mean documents for  more information.
    Note that the Mean of Y given X minimizes L2 Loss of Linear Regression Model.
    Args:
        bias (float, optional): [description]. Defaults to 11.1.
        beta (float, optional): [description]. Defaults to 9.33.
        num_pop (int, optional): [description]. Defaults to 100000000.
        num_features (int, optional): [description]. Defaults to 1.
        pop_error_mean (int, optional): [description]. Defaults to 0.
        pop_error_std_dev (int, optional): [description]. Defaults to 1.
        conditional_x_value (int, optional): [description]. Defaults to 10.
        num_sample (int, optional): [description]. Defaults to 1000000.
    """
    torch.manual_seed(1992)
    np.random.seed(1992)

    device = "cuda"

    x, y, e = generate_linear_regression_data.generate_dataset_simple(
        bias=bias,
        beta=beta,
        num_pop=num_pop,
        num_features=num_features,
        pop_error_mean=pop_error_mean,
        pop_error_std_dev=pop_error_std_dev,
    )

    conditional_x_val = torch.where(x == conditional_x_value)

    print(
        f"\nThere are {conditional_x_val[0].size()} values where x takes on {conditional_x_value}, we are going to find the average values of the corresponding y values to these x.\n"
    )

    conditional_y_on_x_val = y[conditional_x_val]

    conditional_y_on_x_val_mean = torch.mean(conditional_y_on_x_val)

    print(
        f"\nConditional Mean of y value given that x is {conditional_x_value} is {conditional_y_on_x_val_mean}\n"
    )

    real_y_given_x_val = beta * conditional_x_value + bias  # omit error term for now.

    print(
        f"\nThe real value of y value given that x is {conditional_x_value} is {real_y_given_x_val}\n"
    )

    print(
        "\nThis concludes that the Linear Regression Formula coincides with the Conditional Mean of Y given X.\n"
    )

    # Sample

    # Randomly samples num_samples from the population, what I did is just sample num_sample from 0 to num_pop
    # This gives the index of num_samples.

    random_sampling_index = torch.randint(low=0, high=num_pop, size=(num_sample,)).to(device)

    x_sample = x[random_sampling_index].cpu().numpy().reshape(-1, 1)
    y_sample = y[random_sampling_index].cpu().numpy()

    conditional_x_val_sample = np.where(x_sample.flatten() == conditional_x_value)

    conditional_y_on_x_val_sample = y_sample[conditional_x_val_sample]
    conditional_y_on_x_val_sample_mean = np.mean(conditional_y_on_x_val_sample)

    lr_sklearn = LinearRegression(fit_intercept=True, normalize=False).fit(x_sample, y_sample)

    pred_sklearn = lr_sklearn.predict([[conditional_x_value]])

    print(
        f"\nConditional Mean of predicted y value given that x is {conditional_x_value} is {conditional_y_on_x_val_sample_mean}\n"
    )

    print(
        f"\nThe predicted value of y value given that x is {conditional_x_value} using our Linear Regression model with coefficient {lr_sklearn.coef_} and intercept {lr_sklearn.intercept_} is {pred_sklearn[0]}\n"
    )


if __name__ == "__main__":
    regression_diabetes()
