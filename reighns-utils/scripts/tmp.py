import numpy as np
from typing import List
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from numpy import polyfit
from sklearn.pipeline import make_pipeline
from typing import *


np.random.seed(1992)


def f_true(x: np.ndarray) -> np.ndarray:
    """Our ground truth target function.

    shh! This is just a simulation, so we pretend we knew what it is!

    Args:
        x (np.ndarray): [description]

    Returns:
        np.ndarray: [description]
    """
    y = x ** 2
    return y


def generate_sim_data(
    f_true, mu: float = 0, sigma: float = 0.3, has_epsilon=True, sample_size=100
) -> np.ndarray:
    """Here is a function generate simulated data.

    The setup is as follows:

    \mathcal{X} = all points that are generated from the distribution \mathcal{P} = uniform distribution [0, 1]
    \mathcal{\epsilon} = irreducible noise sampled from normal distribution with mean and sigma specified.
    \mathcal{Y} = all points that are the mapping from f_true(\mathcal{X}) over the distribution \mathcal{P}, in other words, any points sampled from here follows a conditional normal distribution Y| X \sim N(f(x), \sigma^2) where we assume \mathcal{X} and \mathcal{\epsilon} are indepedent.
    \mathcal{P} = as mentioned, we specify our distribution to be a uniform distribution [0, 1]

    One confusing point to take note is that this function returns sampled data \mathcal{D} = {X_train, Y_train} where both X_train and Y_train belongs to the true population.


    Args:
        f_true ([type]): The true target function
        mu (float, optional): [description]. Defaults to 0.
        sigma (float, optional): [description]. Defaults to 0.75.
        has_epsilon (bool, optional): [description]. Defaults to False.
        sample_size (int, optional): [description]. Defaults to 100.

    Returns:
        np.ndarray: [description]
    """

    X_train = np.random.uniform(low=0.0, high=1.0, size=sample_size)
    if has_epsilon is True:
        # The real error which is irreducible
        epsilon = np.random.normal(mu, sigma, size=sample_size)
        # Y_train = f_true(X_train) + epsilon
        # why is this not the same as above? Because we need take mu = mean of X_train
        Y_train = np.random.normal(0.81, 0.3, sample_size)
    else:
        Y_train = f_true(X_train)

    return X_train, Y_train


def hypothesis_space(estimator):
    """Just a dummy function to specify our hypothesis space of a model we choose.

    In this example, if we choose a polynomial regression of degree 1, then it is just linear regression.

    Args:
        estimator ([type]): [description]

    Returns:
        [type]: [description]
    """
    return estimator


def get_total_mse(y_true: np.ndarray, y_pred: np.ndarray):
    """get TOTAL mse not MEAN MSE

    \sum_{i=1}^{num_samples}(y_true_i - y_pred_i)^2

    Args:
        y_true (np.ndarray): [description]
        y_pred (np.ndarray): [description]

    Returns:
        [type]: [description]
    """

    total_mse = 0

    for _y_true, _y_pred in zip(y_true, y_pred):
        squared_error = (_y_true - _y_pred) ** 2
        total_mse += squared_error

    assert total_mse == np.sum(np.square(y_true - y_pred))

    return total_mse


def expected_test_error(f_true, estimator, num_simulations, num_samples, Y_test):
    """
    $$
    \widehat{\text{MSE}}\left(f(0.90), \hat{f}_k(0.90)\right) = \frac{1}{n_{\texttt{sims}}}\sum_{i = 1}^{n_{\texttt{sims}}} \left(f(0.90) - \hat{f}_k^{[i]}(0.90) \right)^2
    $$

    """

    num_y_test = Y_test.shape[0]  # num of samples in y_test
    num_simulations = num_simulations  # just for emphasis

    hypothesis_dict: Dict = {}

    for sim in range(1, num_simulations + 1):

        # for simulation 1, then it is h_D_{1}
        curr_hypothesis = f"h_D_{sim}"
        # generated X_train and Y_train from distribution
        X_train, Y_train = generate_sim_data(
            f_true, mu=0, sigma=0.3, has_epsilon=True, sample_size=num_samples
        )
        # fit our hypothesis on X_train, Y_train

        estimator.fit(X_train.reshape(-1, 1), Y_train.reshape(-1, 1))

        # make predictions on X_test = 0.9 (note in our first example we make predictions on one point only)

        prediction_on_X_test = estimator.predict(X_test)

        assert prediction_on_X_test.shape == (num_y_test, 1)

        # basically find mse on the Y_test (ground truth for X_test), and the predictions made on X_test using our current hypothesis fit on the sampled data
        total_mse = get_total_mse(Y_test, prediction_on_X_test)

        # this step will only matter if your number of X_test, Y_test sample is more than 1 point!
        mean_total_mse = total_mse / num_y_test

        # this should look like {"h_{D}_1": 0.6, "h_{D}_2": 0.8, ...}
        hypothesis_dict[curr_hypothesis] = mean_total_mse

    total_test_error = 0
    # Now we loop through the dictionary to find the total test error across ALL HYPOTHESIS we got
    for hypothesis, hypothesis_mean_mse in hypothesis_dict.items():
        total_test_error += hypothesis_mean_mse

    # Finally, average the error over all such hypothesis to get the expected test error!
    expected_test_error = total_test_error / num_simulations

    return expected_test_error


def get_predictions(f_true, estimator, num_simulations, num_samples, Y_test):
    """
    $$
    \widehat{\text{MSE}}\left(f(0.90), \hat{f}_k(0.90)\right) = \frac{1}{n_{\texttt{sims}}}\sum_{i = 1}^{n_{\texttt{sims}}} \left(f(0.90) - \hat{f}_k^{[i]}(0.90) \right)^2
    $$

    """

    num_y_test = Y_test.shape[0]  # num of samples in y_test
    num_simulations = num_simulations  # just for emphasis

    hypothesis_dict: Dict = {}

    for sim in range(1, num_simulations + 1):

        # for simulation 1, then it is h_D_{1}
        curr_hypothesis = f"h_D_{sim}"
        # generated X_train and Y_train from distribution
        X_train, Y_train = generate_sim_data(
            f_true, mu=0, sigma=0.3, has_epsilon=True, sample_size=num_samples
        )
        # fit our hypothesis on X_train, Y_train

        estimator.fit(X_train.reshape(-1, 1), Y_train.reshape(-1, 1))

        # make predictions on X_test = 0.9 (note in our first example we make predictions on one point only)

        prediction_on_X_test = estimator.predict(X_test)

        assert prediction_on_X_test.shape == (num_y_test, 1)

        # this should look like {"h_{D}_1": 0.6, "h_{D}_2": 0.8, ...}

        hypothesis_dict[curr_hypothesis] = prediction_on_X_test

    all_preds = np.ones(
        shape=(num_simulations, num_y_test)
    )  # create matrix of num_sim x num_y_test size where each row is the predictions of y_test for hypothesis_i
    for index, (hypothesis, hypothesis_predictions) in enumerate(hypothesis_dict.items()):
        all_preds[index] = hypothesis_predictions

    return hypothesis_dict, all_preds


if __name__ == "__main__":
    np.random.seed(1992)
    # we choose 4 Hypothesis Set for comparison. Please note this represents 4 different models of choice we choose. In order to avoid confusion, we will try to call the "models" in each Hypothesis Set "hypothesis".
    num_simulations = 250
    num_samples = 100
    X_test = np.array([[0.9]])  # 0.9 Note carefully we are using only 1 and only 1 test point here.
    # 0.81 note here has no noise as we are not technically generating from the distribution.
    Y_test = f_true(X_test)
    num_y_test = Y_test.shape[0]

    # h_{\theta}(x) = \theta_{n}(x^n) + \theta_{n-1}(x^(n-1)) + ... + \theta_{0} = \sum_{j=0}^{n} \theta_{j}x^{j}

    polynomial_degree_1 = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())

    # expected_test_error_p1 = expected_test_error(
    #     f_true=f_true,
    #     estimator=polynomial_degree_1,
    #     num_simulations=num_simulations,
    #     num_samples=num_samples,
    #     Y_test=Y_test,
    # )

    all_predictions_dict, all_predictions = get_predictions(
        f_true=f_true,
        estimator=polynomial_degree_1,
        num_simulations=num_simulations,
        num_samples=num_samples,
        Y_test=Y_test,
    )

    assert all_predictions.shape == (num_simulations, num_y_test)
    # our h_bar or \bar{h} - which plays an important role in Bias-Variance Tradeoff
    average_hypothesis_predictions = np.mean(all_predictions, axis=0)

    # Expected MSE(f_true(0.9), h_bar(0.9)) = \dfrac{1}{n_sim} \sum_{i=1}^{n_sims} 1/num_y_test (f_true(0.9) - h_bar(0.9)) ** 2
    total_test_error = 0
    for i in range(num_simulations):
        total_squared_error_for_current_hypothesis = (Y_test - all_predictions[i]) ** 2
        total_squared_error_for_current_hypothesis = (
            total_squared_error_for_current_hypothesis / num_y_test
        )
        total_test_error += total_squared_error_for_current_hypothesis

    expected_test_error = total_test_error / num_simulations

    bias = average_hypothesis_predictions - Y_test
    variance = np.sum((all_predictions - average_hypothesis_predictions) ** 2) / num_simulations
    print(expected_test_error)
    print(bias ** 2)
    print(variance)
    print(bias ** 2 + variance)
