import numpy as np
from typing import List
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(1992)


# def true_function(
#     num_population: int = 1000,
#     num_features: int = 1,
# ):
#     """[summary]

#     Args:
#         num_population (int, optional): [description]. Defaults to 1000.
#         num_features (int, optional): [description]. Defaults to 1.

#     Returns:
#         [type]: [description]
#     """

#     # X: our population is days here.
#     X_true = np.arange(1, num_population + 1, 1).reshape(num_population, num_features)

#     # Here we use interpl1d to generate a smooth cubic function

#     tmp_Y = X_true.flatten() ** 2

#     # f: Construct the true function f
#     f_true = interp1d(X_true.flatten(), tmp_Y, kind="cubic")

#     # Y: our corresponding Y
#     Y_true = f_true(X_true)

#     return f_true, X_true, Y_true


def true_function(x):
    """[summary]

    Args:
        num_population (int, optional): [description]. Defaults to 1000.
        num_features (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """

    return x ** 2


def generate_sim_data(f_true, sample_size=100):
    mu, sigma = 0, 0.75  # mean, std

    # The real error which is irreducible
    epsilon = np.random.normal(mu, sigma, size=1)
    X_true = np.random.uniform(low=0.0, high=1.0, size=sample_size)
    Y_true = true_function(X_true) + epsilon

    return f_true, X_true, Y_true, epsilon


def varied_model_variance(
    model,
    f_true: true_function,
    X_true: np.ndarray = None,
    Y_true: np.ndarray = None,
    X_train: np.ndarray = None,
    Y_train: np.ndarray = None,
    X_test: np.ndarray = None,
    Y_test: np.ndarray = None,
    num_simulations: int = 100,
    num_samples: int = 100,
    num_features: int = 1,
    replace: bool = True,
) -> float:

    # This should hold all predictions of K number of hypothesis, or in this case, the number of simulations is K
    all_h_predictions: List[any] = []

    for _ in range(1, num_simulations + 1):
        # In each simulation, sample data X_sampled from our X_true
        X_sampled = np.random.choice(X_true.flatten(), replace=replace, size=num_samples).reshape(
            num_samples, num_features
        )

        # We get the corresponding Y_true with our f_true function
        Y_sampled = f_true(X_sampled)

        # Fit the model with the sampled data
        _ = model.fit(X_sampled, Y_sampled)

        # This forms a hypothesis h_{i} and we use it to predict on the test set
        h_i_pred = model.predict(X_test)
        # print(h_i_pred)

        all_h_predictions.append(h_i_pred)

    # This is only for prediction of one single point

    mean_hypothesis_pred = np.mean(all_h_predictions, axis=0)
    # print(f"Mean Hypothesis Best Prediction is {mean_hypothesis_pred}")

    total_var = 0

    for h_pred in all_h_predictions:
        # print(h_pred, h_pred - mean_hypothesis_pred)
        summand = (h_pred - mean_hypothesis_pred) ** 2
        # print(total_var)
        total_var += summand

    avg_var = total_var / num_simulations

    return avg_var


def varied_model_bias(
    model,
    f_true: true_function,
    X_true: np.ndarray = None,
    Y_true: np.ndarray = None,
    X_train: np.ndarray = None,
    Y_train: np.ndarray = None,
    X_test: np.ndarray = None,
    Y_test: np.ndarray = None,
    num_simulations: int = 100,
    num_samples: int = 100,
    num_features: int = 1,
    replace: bool = True,
) -> float:

    # This should hold all predictions of K number of hypothesis, or in this case, the number of simulations is K
    all_h_predictions: List[any] = []

    for _ in range(1, num_simulations + 1):
        # In each simulation, sample data X_sampled from our X_true
        X_sampled = np.random.choice(X_true.flatten(), replace=replace, size=num_samples).reshape(
            num_samples, num_features
        )

        # We get the corresponding Y_true with our f_true function
        Y_sampled = f_true(X_sampled)

        # Fit the model with the sampled data
        _ = model.fit(X_sampled, Y_sampled)

        # This forms a hypothesis h_{i} and we use it to predict on the test set
        h_i_pred = model.predict(X_test)
        # print(h_i_pred)

        all_h_predictions.append(h_i_pred)

    # This is only for prediction of one single point

    mean_hypothesis_pred = np.mean(all_h_predictions, axis=0)
    # print(f"Mean Hypothesis Best Prediction is {mean_hypothesis_pred}")

    average_hypothesis_prediction = 0

    for h_pred in all_h_predictions:
        # print(h_pred, h_pred - mean_hypothesis_pred)
        summand = h_pred

        average_hypothesis_prediction += summand
        # print(average_hypothesis_prediction)

    average_hypothesis = average_hypothesis_prediction / num_simulations
    print("avg hypothesis", average_hypothesis)
    avg_bias = average_hypothesis - Y_true

    return avg_bias


def estimate_mse(
    model,
    f_true: true_function,
    X_true: np.ndarray = None,
    Y_true: np.ndarray = None,
    X_train: np.ndarray = None,
    Y_train: np.ndarray = None,
    X_test: np.ndarray = None,
    Y_test: np.ndarray = None,
    num_simulations: int = 100,
    num_samples: int = 100,
    num_features: int = 1,
    replace: bool = True,
):
    # This should hold all predictions of K number of hypothesis, or in this case, the number of simulations is K
    all_h_predictions: List[any] = []

    for _ in range(1, num_simulations + 1):
        # In each simulation, sample data X_sampled from our X_true
        X_sampled = np.random.choice(X_true.flatten(), replace=replace, size=num_samples).reshape(
            num_samples, num_features
        )

        # We get the corresponding Y_true with our f_true function
        Y_sampled = f_true(X_sampled)

        # Fit the model with the sampled data
        _ = model.fit(X_sampled, Y_sampled)

        # This forms a hypothesis h_{i} and we use it to predict on the test set
        h_i_pred = model.predict(X_test)
        # print(h_i_pred)

        all_h_predictions.append(h_i_pred)

    summand = 0
    for h_i_pred in all_h_predictions:
        summand += (Y_true - h_i_pred) ** 2

    mse_hat = summand / num_simulations
    return mse_hat


if __name__ == "__main__":

    f_true, X_true, Y_true, epsilon = generate_sim_data(f_true=true_function, sample_size=100)
    print(X_true.shape, Y_true.shape)
    # Randomly sample 1 test point X_test_single_point and reshape it to
    # X_test_single_point = np.random.choice(X_true.flatten(), size=1, replace=False).reshape(-1, 1)
    X_test_single_point = np.array([0.9], dtype=float).reshape(-1, 1)
    Y_test_single_point = f_true(X_test_single_point)
    print(X_test_single_point, Y_test_single_point)

    # Define our Hypothesis to be Linear Regression Simple

    h = LinearRegression()

    # y_pred_single_point = h.predict(X_test_single_point)
    # print("MSE", mean_squared_error(Y_test_single_point, y_pred_single_point))

    mse_hat = estimate_mse(
        model=h,
        f_true=f_true,
        X_true=X_true,
        Y_true=Y_test_single_point,
        X_test=X_test_single_point,
        num_simulations=200,
        num_samples=100,
        num_features=1,
        replace=True,
    )
    print("MSE hat", mse_hat)
    var = varied_model_variance(
        model=h,
        f_true=f_true,
        X_true=X_true,
        X_test=X_test_single_point,
        num_simulations=200,
        num_samples=100,
        num_features=1,
        replace=True,
    )

    print("VAR", var)

    bias = varied_model_bias(
        model=h,
        f_true=f_true,
        X_true=X_true,
        Y_true=Y_test_single_point,
        X_test=X_test_single_point,
        num_simulations=200,
        num_samples=100,
        num_features=1,
        replace=True,
    )
    print("bias", bias)

    print(epsilon)
    print(bias ** 2 + var)
