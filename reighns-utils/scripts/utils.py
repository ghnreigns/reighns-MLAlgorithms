import functools

from sklearn.exceptions import NotFittedError


def NotFitted(func):
    """A decorator such that NotFittedError will be raised whenever an user tries to execute a Linear Regression Model without calling `.fit()` first.
    Args:
        func ([type]): [description]
    Raises:
        NotFittedError: [description]
    Returns:
        [type]: [description]
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._fitted:
            # print(self._fitted)
            raise NotFittedError  # or you can define custom error NotFittedError()
        else:
            # print(self._fitted)
            return func(self, *args, **kwargs)

    return wrapper
