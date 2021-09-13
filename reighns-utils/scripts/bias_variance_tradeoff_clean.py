import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from typing import Dict


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
    f_true, mu: float = 0, sigma: float = 0.1, has_epsilon=True, sample_size=100
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
        Y_train = f_true(X_train) + epsilon
        # why is this not the same as above? Because we need take mu = mean of X_train
        # Y_train = np.random.normal(0.81, 0.3, sample_size)
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


def get_predictions(f_true, estimator, num_simulations, num_samples, X_test, Y_test):
    """
    $$
    \widehat{\text{MSE}}\left(f(0.90), \hat{f}_k(0.90)\right) = \frac{1}{n_{\texttt{sims}}}\sum_{i = 1}^{n_{\texttt{sims}}} \left(f(0.90) - \hat{f}_k^{[i]}(0.90) \right)^2
    $$

    """

    num_y_test = Y_test.shape[0]  # num of samples in y_test
    # num_simulations = num_simulations  # just for emphasis

    hypothesis_dict: Dict = {}

    for sim in range(1, num_simulations + 1):

        # for simulation 1, then it is h_D_{1}
        curr_hypothesis = f"h_D_{sim}"
        # generated X_train and Y_train from distribution
        X_train, Y_train = generate_sim_data(
            f_true, mu=0, sigma=0.1, has_epsilon=True, sample_size=num_samples
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

        hypothesis_predictions = hypothesis_predictions.reshape(-1)
        # print(hypothesis_predictions.shape)
        # print(all_preds[index].shape)
        all_preds[index] = hypothesis_predictions

    return hypothesis_dict, all_preds


def mean_squared_error(y_true, y_pred):
    """Just calculate MEAN SQUARED ERROR

    Args:
        y_true ([type]): [description]
        y_pred ([type]): [description]

    Returns:
        [type]: [description]
    """

    mse = np.sum(np.square(y_true - y_pred)) / y_pred.shape[0]

    return mse


def get_expected_test_error(Y_test, all_predictions, num_simulations):
    """[summary]

    Y_test = [[0.81], [1.44]]
    all_predictions = [h1, h2, h3, ..., h_k]


    Args:
        h_bar ([type]): [description]
        all_predictions ([type]): [description]
        num_simulations ([type]): [description]
        num_y_test ([type]): [description]
    """

    total_test_error = 0
    for i in range(num_simulations):
        curr_hypothesis_predictions = all_predictions[i].reshape(Y_test.shape)

        mean_squared_error_for_current_hypothesis = mean_squared_error(
            Y_test, curr_hypothesis_predictions
        )

        total_test_error += mean_squared_error_for_current_hypothesis

    expected_test_error = total_test_error / num_simulations
    return expected_test_error


def get_bias(Y_test, h_bar):
    """[summary]

    all_predictions = [h1, h2, h3, ..., h_k]
    h_bar = 1/k (h1+h2+...+h_k)

    Args:
        h_bar ([type]): [description]
        all_predictions ([type]): [description]
        num_simulations ([type]): [description]
        num_y_test ([type]): [description]
    """

    h_bar = h_bar.reshape(Y_test.shape)
    bias = np.sum(np.square(h_bar - Y_test)) / Y_test.shape[0]
    assert bias == mean_squared_error(h_bar, Y_test)
    return bias


def get_variance(Y_test, all_predictions, h_bar, num_simulations):

    total_error_deviated_from_average_hypothesis = 0
    for i in range(num_simulations):
        curr_hypothesis_predictions = all_predictions[i].reshape(Y_test.shape)
        total_mean_squared_error_for_current_hypothesis_vs_average_hypothesis = mean_squared_error(
            h_bar, curr_hypothesis_predictions
        )

        total_error_deviated_from_average_hypothesis += (
            total_mean_squared_error_for_current_hypothesis_vs_average_hypothesis
        )

    variance = total_error_deviated_from_average_hypothesis / num_simulations
    return variance


if __name__ == "__main__":

    # only using on one point
    np.random.seed(1992)

    num_simulations = 2
    num_samples = 3
    X_test = np.array([[0.9], [1.2]])
    # X_test = np.array([[0.9]])

    Y_test = f_true(X_test)

    num_y_test = Y_test.shape[0]

    polynomial_degree_1 = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())

    # I decided to follow STAT432 style and get all predictions from various hypothesis first as it is cleaner.
    all_predictions_dict, all_predictions = get_predictions(
        f_true=f_true,
        estimator=polynomial_degree_1,
        num_simulations=num_simulations,
        num_samples=num_samples,
        Y_test=Y_test,
        X_test=X_test,
    )

    h_bar = np.mean(all_predictions, axis=0).reshape(Y_test.shape)

    assert h_bar.shape == Y_test.shape == (num_y_test, 1)
    assert all_predictions.shape == (num_simulations, num_y_test)

    expected_test_error = get_expected_test_error(Y_test, all_predictions, num_simulations)

    print("Expected Test Error", expected_test_error)

    bias = get_bias(Y_test, h_bar)
    print("Bias", bias)

    variance = get_variance(Y_test, all_predictions, h_bar, num_simulations)
    print("Variance", variance)

    print("Bias Squared + Variance", bias + variance)
