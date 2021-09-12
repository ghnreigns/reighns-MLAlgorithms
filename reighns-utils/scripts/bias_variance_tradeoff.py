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


def expected_test_error(f_true, estimator, num_simulations, num_samples, Y_test, X_test):
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

        hypothesis_predictions = hypothesis_predictions.reshape(-1)
        # print(hypothesis_predictions.shape)
        # print(all_preds[index].shape)
        all_preds[index] = hypothesis_predictions

    return hypothesis_dict, all_preds


if __name__ == "__main__":

    def one_test_point():
        # only using on one point
        np.random.seed(1992)
        # we choose 4 Hypothesis Set for comparison. Please note this represents 4 different models of choice we choose. In order to avoid confusion, we will try to call the "models" in each Hypothesis Set "hypothesis".
        num_simulations = 250
        num_samples = 100
        X_test = np.array(
            [[0.9]]
        )  # 0.9 Note carefully we are using only 1 and only 1 test point here.

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
        #     X_test= X_test
        # )

        # I decided to follow STAT432 style and get all predictions from various hypothesis first as it is cleaner.
        all_predictions_dict, all_predictions = get_predictions(
            f_true=f_true,
            estimator=polynomial_degree_1,
            num_simulations=num_simulations,
            num_samples=num_samples,
            Y_test=Y_test,
            X_test=X_test,
        )

        assert all_predictions.shape == (num_simulations, num_y_test)
        # our h_bar or \bar{h} - which plays an important role in Bias-Variance Tradeoff
        average_hypothesis_predictions = np.mean(all_predictions, axis=0)
        print("average hypothesis", average_hypothesis_predictions)

        # Expected MSE(f_true(0.9), h_bar(0.9)) = \dfrac{1}{n_sim} \sum_{i=1}^{n_sims} 1/num_y_test (f_true(0.9) - h_bar(0.9)) ** 2
        total_test_error = 0
        for i in range(num_simulations):
            total_squared_error_for_current_hypothesis = (
                Y_test - all_predictions[i].reshape(Y_test.shape)
            ) ** 2

            # rmb to divide by num of points in y_test
            total_squared_error_for_current_hypothesis = (
                total_squared_error_for_current_hypothesis / num_y_test
            )
            total_test_error += total_squared_error_for_current_hypothesis

        expected_test_error = total_test_error / num_simulations

        # Bias : $$
        #        \widehat{\text{bias}} \left(\hat{f}(0.90) \right)  = \frac{1}{n_{\texttt{sims}}}\sum_{i = 1}^{n_{\texttt{sims}}} \left(\hat{f}_k^{[i]}(0.90) \right) - f(0.90)
        #        $$
        # This is just the deviation of our average hypothesis predictions from the ground truth Y_test
        # Remember to square it!
        bias = average_hypothesis_predictions.reshape(Y_test.shape) - Y_test

        # Variance:
        # $$
        # \widehat{\text{var}} \left(\hat{f}(0.90) \right) = \frac{1}{n_{\texttt{sims}}}\sum_{i = 1}^{n_{\texttt{sims}}} \left(\hat{f}_k^{[i]}(0.90) - \frac{1}{n_{\texttt{sims}}}\sum_{i = 1}^{n_{\texttt{sims}}}\hat{f}_k^{[i]}(0.90) \right)^2
        # $$
        # This intuitively just illustrates for each prediction made by each hypothesis, how much are they deviating from the average hypothesis predictions?
        # To quantify this,
        # 1. total_error_deviated_from_average_hypothesis = add up all deviations from h_i from \bar{h}

        total_error_deviated_from_average_hypothesis = 0
        for i in range(num_simulations):
            total_squared_error_for_current_hypothesis_vs_average_hypothesis = (
                all_predictions[i] - average_hypothesis_predictions
            ) ** 2
            total_squared_error_for_current_hypothesis_vs_average_hypothesis = (
                total_squared_error_for_current_hypothesis_vs_average_hypothesis / num_y_test
            )
            total_error_deviated_from_average_hypothesis += (
                total_squared_error_for_current_hypothesis_vs_average_hypothesis
            )

        variance = total_error_deviated_from_average_hypothesis / num_simulations

        # variance = np.sum((all_predictions - average_hypothesis_predictions) ** 2) / num_simulations
        print(expected_test_error)
        print(bias ** 2)
        print(variance)
        print(bias ** 2 + variance)  # see this matches perfectly with the decomposition.

    def two_test_point():
        # only using on one point
        np.random.seed(1992)
        # we choose 4 Hypothesis Set for comparison. Please note this represents 4 different models of choice we choose. In order to avoid confusion, we will try to call the "models" in each Hypothesis Set "hypothesis".
        num_simulations = 2
        num_samples = 3
        X_test = np.array(
            [[0.9], [1.2]]
        )  # 0.9 Note carefully we are using only 1 and only 1 test point here.

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

        # I decided to follow STAT432 style and get all predictions from various hypothesis first as it is cleaner.
        all_predictions_dict, all_predictions = get_predictions(
            f_true=f_true,
            estimator=polynomial_degree_1,
            num_simulations=num_simulations,
            num_samples=num_samples,
            Y_test=Y_test,
            X_test=X_test,
        )

        assert all_predictions.shape == (num_simulations, num_y_test)
        # our h_bar or \bar{h} - which plays an important role in Bias-Variance Tradeoff

        average_hypothesis_predictions = np.mean(all_predictions, axis=0)
        print("Average Hypothesis", average_hypothesis_predictions)

        # Expected MSE(f_true(0.9), h_bar(0.9)) = \dfrac{1}{n_sim} \sum_{i=1}^{n_sims} 1/num_y_test (f_true(0.9) - h_bar(0.9)) ** 2
        total_test_error = 0
        for i in range(num_simulations):
            # Y_test = [[0.81], [1.44]]
            # all_predictions[0] = [[0.9], [1.4]]
            # total_squared_error_for_current_hypothesis = (Y_test - all_predictions[0]) ** 2 = [[-0.09], [0.04]] ** 2 = [[0.081], [0.016]]
            # 0.081 is the error on the first point, whereas 0.016 is the error on the second point.
            # Thus you need to sum them up by calling np.sum()

            total_squared_error_for_current_hypothesis = np.sum(
                (Y_test - all_predictions[i].reshape(Y_test.shape)) ** 2
            )
            # rmb to divide by num of points in y_test
            total_squared_error_for_current_hypothesis = (
                total_squared_error_for_current_hypothesis / num_y_test
            )
            total_test_error += total_squared_error_for_current_hypothesis

        expected_test_error = total_test_error / num_simulations

        # Bias : $$
        #        \widehat{\text{bias}} \left(\hat{f}(0.90) \right)  = \frac{1}{n_{\texttt{sims}}}\sum_{i = 1}^{n_{\texttt{sims}}} \left(\hat{f}_k^{[i]}(0.90) \right) - f(0.90)
        #        $$
        # This is just the deviation of our average hypothesis predictions from the ground truth Y_test
        # Remember to square it!

        # Same example:
        # average_hypothesis_predictions = [[0.80349123], [0.79944328]]
        # Y_test = [[0.81], [1.44]]
        # Bias for each prediction = [[-0.0065], [-0.64]]
        # Bias =  absolute error for average prediction vs the ground truth Y_test so you just add the error up for each prediction
        # Bias = np.sum(average_hypothesis_predictions - Y_test)

        bias = (
            np.sum((average_hypothesis_predictions.reshape(Y_test.shape) - Y_test) ** 2)
            / num_y_test
        )

        # Variance:
        # $$
        # \widehat{\text{var}} \left(\hat{f}(0.90) \right) = \frac{1}{n_{\texttt{sims}}}\sum_{i = 1}^{n_{\texttt{sims}}} \left(\hat{f}_k^{[i]}(0.90) - \frac{1}{n_{\texttt{sims}}}\sum_{i = 1}^{n_{\texttt{sims}}}\hat{f}_k^{[i]}(0.90) \right)^2
        # $$
        # This intuitively just illustrates for each prediction made by each hypothesis, how much are they deviating from the average hypothesis predictions?
        # To quantify this,
        # 1. total_error_deviated_from_average_hypothesis = add up all deviations from h_i from \bar{h}

        total_error_deviated_from_average_hypothesis = 0
        for i in range(num_simulations):
            total_squared_error_for_current_hypothesis_vs_average_hypothesis = np.sum(
                (all_predictions[i] - average_hypothesis_predictions) ** 2
            )

            total_squared_error_for_current_hypothesis_vs_average_hypothesis = (
                total_squared_error_for_current_hypothesis_vs_average_hypothesis / num_y_test
            )
            total_error_deviated_from_average_hypothesis += (
                total_squared_error_for_current_hypothesis_vs_average_hypothesis
            )

        variance = total_error_deviated_from_average_hypothesis / num_simulations

        # variance = np.sum((all_predictions - average_hypothesis_predictions) ** 2) / num_simulations
        print("Expected Test Error", expected_test_error)
        print("Bias Squared", bias)
        print("Variance", variance)
        print(
            "Expected Test Error = Bias Squared + Variance = ", bias + variance
        )  # see this matches perfectly with the decomposition.

    def two_test_point_cleaned():
        # only using on one point
        np.random.seed(1992)

        num_simulations = 2
        num_samples = 3
        X_test = np.array([[0.9], [1.2]])
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

        # Y_test = [[0.81], [1.44]]
        # X_test = [[0.9], [1.2]]
        # We ran 2 simulations, so we have 2 hypothesis
        # h1 = [1.07657572 1.13156252]
        # h2 = [3.08344394 4.1038925 ]
        # Stacking together we have
        # h1 and h2 = [[1.07657572 1.13156252]
        #               [3.08344394 4.1038925 ]]

        # h_bar = [2.08000983 2.61772751] which is gotten by (h1 + h2)/2
        # Note h_bar's shape is same as Y_test shape, h_bar[0] is average best prediction for 0.9 and h_bar[1] is best prediction for 1.2

        assert all_predictions.shape == (num_simulations, num_y_test)

        average_hypothesis_predictions = np.mean(all_predictions, axis=0)
        print("Average Hypothesis", average_hypothesis_predictions)

        ### Expected Test Error ###
        # for each hypothesis, we use this hypothesis to predict on X_test, we get h1 first, and we find the mean squared difference of hi and Y_test
        # MSE(h1, Y_test) = np.sum([[0.07106262], [0.09513368]]) =  0.1661962942345044 where 0.07106262 is the mean squared error of 0.81 and 1.0765, 0.09513368 is the mean squared error of 1.44 and 1.13156252
        # MEAN_MSE(h1, Y_test) = 0.1661962942345044 / 2 = 0.0830981471172522
        # MSE(h2, Y_test) = np.sum([[5.16854734], [7.09632328]]) =  12.264870610777585
        # MEAN_MSE(h2, Y_test) = 12.264870610777585 / 2 = 6.132435305388793

        # Now we have both mean MSE for h1 and h2 with respect to Y_test
        # We add both up and divide the total MSE by the number of simulations we have, as we are taking expectation of possible D
        # Expected MSE = (0.0830981471172522 + 6.132435305388793) / num_simulations = 3.1077667262530224

        total_test_error = 0
        for i in range(num_simulations):
            total_squared_error_for_current_hypothesis = np.sum(
                (Y_test - all_predictions[i].reshape(Y_test.shape)) ** 2
            )

            total_squared_error_for_current_hypothesis = (
                total_squared_error_for_current_hypothesis / num_y_test
            )
            total_test_error += total_squared_error_for_current_hypothesis

        expected_test_error = total_test_error / num_simulations

        ### Bias ###

        # Recall h_bar = [2.08000983 2.61772751]
        # We want to calculate the mean squared error of h_bar and Y_test
        # And what is the definition of MEAN squared error again? It is the mean of the squared errors, so please do not forget to divide the number of y_test after summing the squared error.
        # (h_bar - Y_test)**2 = [[1.61292497], [1.3870421]] means bias for point 0.9 is 1.6129 and for point 1.2 is 1.387
        # We then sum the above to get the squared error, and divide by the number of points we have to get the MEAN squared error

        bias = (
            np.sum((average_hypothesis_predictions.reshape(Y_test.shape) - Y_test) ** 2)
            / num_y_test
        )

        ### Variance ###

        # Recall h_bar = [2.08000983 2.61772751]
        # We want to calculate the mean squared error of h_bar and h1, h2, h3
        # MSE(h1, h_bar) = 1.6077...
        # MSE(h2, h_bar) = 1.6077...

        # Add up all the MSE(hi, h_bar) = sum(h1, h2, ...)
        # Lastly, divide by the number of simulations.

        total_error_deviated_from_average_hypothesis = 0
        for i in range(num_simulations):

            total_squared_error_for_current_hypothesis_vs_average_hypothesis = np.sum(
                (all_predictions[i] - average_hypothesis_predictions) ** 2
            )

            total_squared_error_for_current_hypothesis_vs_average_hypothesis = (
                total_squared_error_for_current_hypothesis_vs_average_hypothesis / num_y_test
            )

            total_error_deviated_from_average_hypothesis += (
                total_squared_error_for_current_hypothesis_vs_average_hypothesis
            )

        variance = total_error_deviated_from_average_hypothesis / num_simulations

        # variance = np.sum((all_predictions - average_hypothesis_predictions) ** 2) / num_simulations
        print("Expected Test Error", expected_test_error)
        print("Bias Squared", bias)
        print("Variance", variance)
        print(
            "Expected Test Error = Bias Squared + Variance = ", bias + variance
        )  # see this matches perfectly with the decomposition.

    one_test_point()
    two_test_point_cleaned()
