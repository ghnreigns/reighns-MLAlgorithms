import numpy as np


def true_false_positive(y_pred_thresholded: np.array, y_true: np.array):
    """
    Returns the tpr and fpr rate. This is a simple 2 class confusion matrix, can be extended to multi class, but start from this first.
    1 is pos class, 0 is neg class.

            Parameters:
                    y_pred_thresholded (np.array): A decimal integer

                    y_true (np.array): Another decimal integer

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
    true_positive = np.equal(y_pred_thresholded, 1) & np.equal(y_true, 1)
    true_negative = np.equal(y_pred_thresholded, 0) & np.equal(y_true, 0)
    false_positive = np.equal(y_pred_thresholded, 1) & np.equal(y_true, 0)
    false_negative = np.equal(y_pred_thresholded, 0) & np.equal(y_true, 1)

    tpr = true_positive.sum() / (true_positive.sum() + false_negative.sum())
    fpr = false_positive.sum() / (false_positive.sum() + true_negative.sum())

    return tpr, fpr


def roc_from_scratch(y_pred: np.array, y_true: np.array, partitions=100):
    """
    This function `roc_from_scratch` takes in `y_pred` and `y_true` and a default argument `partitions = 100` and returns an `np.array` of shape `(partitions+1, 2)`.
    This function iterates over `partitions+1` times. As an example, if we take `partitions = 10`, then we start from 0, and increment 1 until it reaches 10, a total of 11 times.
    In the `for` loop, we calculate the `fpr` and `tpr` of `y_pred` and `y_true` at each `threshold_value`, given by `i/partitions`.
    So in the first loop and final loop, we will always take the threshold value of 0 and 1, to ensure our both ends starts and end at (0,0) and (1,1) respectively.

            Parameters:
                    y_pred (np.array): predictions of the dataset in raw logits/probability form

                    y_true (np.array): ground truth of the dataset in 0 and 1 form for binary

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
                    >>> y_true = np.array([0, 0, 1, 1])
                    >>> y_pred = np.array([0.1, 0.4, 0.35, 0.8])
                    >>> roc_from_scratch(y_pred, y_true,partitions=10)
                    array([ [1. , 1. ],
                            [1. , 1. ],
                            [0.5, 1. ],
                            [0.5, 1. ],
                            [0.5, 0.5],
                            [0. , 0.5],
                            [0. , 0.5],
                            [0. , 0.5],
                            [0. , 0.5],
                            [0. , 0. ],
                            [0. , 0. ] ])
    """
    tpr_fpr_array = np.array([])

    for i in range(partitions + 1):
        threshold_value = i / partitions
        y_pred_thresholded = np.greater_equal(y_pred, threshold_value).astype(int)

        tpr, fpr = true_false_positive(y_pred_thresholded, y_true)
        tpr_fpr_array = np.append(tpr_fpr_array, [fpr, tpr])
        tpr_fpr_array = tpr_fpr_array.reshape((-1, 2))
    return tpr_fpr_array


def auc(fpr: np.array, tpr: np.array):
    """
    This function takes in fpr and tpr and calculates the integral of the tpr vs fpr graph using composite trapezoidal rule.
    We take y = tpr and x = fpr. We further take note that fpr and tpr on the x and y axis must be given in a monotone increasing/decreasing manner in a syncrhonized movement.
    Since it is integration, we can also use rectangles to estimate the area under the graph, where `dx` is the interval between each adjacent x-value (fpr).
    The height is given by the value of y at the point x. We note that we can use `np.diff` to check if the `dx` values are monotone.
    If `dx` are all negative, then it indicates the area is negative and we need to multiply by -1.

            Parameters:
                    fpr (np.array): predictions of the dataset in raw logits/probability form

                    tpr (np.array): ground truth of the dataset in 0 and 1 form for binary

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
                    >>> y_true = np.array([0, 0, 1, 1])
                    >>> y_pred = np.array([0.1, 0.4, 0.35, 0.8])
                    >>> roc_from_scratch(y_pred, y_true,partitions=10)
                    array([ [1. , 1. ],
                            [1. , 1. ],
                            [0.5, 1. ],
                            [0.5, 1. ],
                            [0.5, 0.5],
                            [0. , 0.5],
                            [0. , 0.5],
                            [0. , 0.5],
                            [0. , 0.5],
                            [0. , 0. ],
                            [0. , 0. ] ])
    """
    direction = 1
    dx = np.diff(fpr)
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            raise ValueError(
                "x is neither increasing nor decreasing " ": {}.".format(fpr)
            )
    area = np.trapz(y=tpr, x=fpr)
    return area


if __name__ == "__main__":
    #!pip install celluloid
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt
    import seaborn as sns

    np.random.seed(1930)

    """
    1000 rows of training data;
    20 features (think of tumor size, tumor color etc to predict malignancy)
    2 classes - think of malignant and benign
    """
    X, y = make_classification(
        n_samples=1000,
        n_informative=10,
        n_features=20,
        flip_y=0.2,
        random_state=1930,
        n_classes=2,
    )
    # print(X.shape)
    # print(set(y))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1930
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    prob_vector = model.predict_proba(X_test)[:, 1]
    # returns a 1000 by 1 array, with single probs indicating the probs of the class being class 1 (positive class)
    # print(prob_vector)
