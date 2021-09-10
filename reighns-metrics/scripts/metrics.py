import numpy as np


def unhot(function):
    """Convert one-hot representation into one column."""

    def wrapper(actual, predicted):
        if len(actual.shape) > 1 and actual.shape[1] > 1:
            actual = actual.argmax(axis=1)
        if len(predicted.shape) > 1 and predicted.shape[1] > 1:
            predicted = predicted.argmax(axis=1)
        return function(actual, predicted)

    return wrapper


@unhot
def accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    """Accuracy rate.

    Args:
        y_true (np.ndarray): [description]
        y_pred (np.ndarray): [description]

    Returns:
        [type]: [description]
    """
    return 1.0 - classification_error(y_true, y_pred)


@unhot
def classification_error(y_true: np.ndarray, y_pred: np.ndarray):
    """Error rate.

    Args:
        y_true (np.ndarray): [description]
        y_pred (np.ndarray): [description]

    Returns:
        [type]: [description]
    """

    return (y_true != y_pred).sum() / float(y_true.shape[0])
