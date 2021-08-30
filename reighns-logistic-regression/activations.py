import numpy as np


class Sigmoid:
    """
    Implements the sigmoid function. Suggestive parameter z = w^Tx+b

            Parameters:
                    z: z = w^Tx + b

            Returns:
                    tpr (float):
                            True positive rate (TPR) where given a threshold t, and an input of y_pred_thresholded vector of
                            1s and 0s based on the decision rule y_pred > t implies positive class.

                    fpr (float):
                            False positive rate (FPR) where given a threshold t, and an input of y_pred_thresholded vector of
                            1s and 0s based on the decision rule y_pred > t implies positive class.
            Examples:
            --------
                    >>> import numpy as np
                    >>> from sklearn import metrics
                    >>> y_true = np.array([1,1,0,1,0,0])
                    >>> y_pred_thresholded = np.array([1,1,1,0,0,0])
                    >>> fpr, tpr = true_false_positive(y_pred_thresholded, y_true)
                    >>> fpr, tpr
                    0.66, 0.33
    """

    def __call__(self, x):
        g = 1 / (1 + np.exp(-x))
        return g

    def gradient(self, x):
        g = self.__call__(x)
        dg_dx = g * (1 - g)
        return dg_dx
