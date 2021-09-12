import numpy as np


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-12):
    """
        https://stackoverflow.com/questions/47377222/what-is-the-problem-with-my-implementation-of-the-cross-entropy-function
        Computes cross entropy between targets (encoded as one-hot vectors)
        and y_pred.
        Input: y_pred (N, k) ndarray
               y_true (N, k) ndarray
        Returns: scalar
        predictions = np.array([[0.25,0.25,0.25,0.25],
                            [0.01,0.01,0.01,0.96]])
    targets = np.array([[0,0,0,1],
                       [0,0,0,1]])
                       ans = 0.71355817782  #Correct answer
    x = cross_entropy(predictions, targets)
    print(np.isclose(x,ans))
    """
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    # take note that y_pred is of shape 1 x n_samples as stated in our framework
    n_samples = y_pred.shape[1]

    # cross entropy function
    cross_entropy_function = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    # cross entropy function here is same shape as y_true and y_pred since we are
    # just performing element wise operations on both of them.
    assert cross_entropy_function.shape == (1, n_samples)

    # we sum up all the loss for each individual sample
    total_cross_entropy_loss = -np.sum(cross_entropy_function, axis=1)
    assert total_cross_entropy_loss.shape == (1,)

    # we then average out the total loss across m samples, but we squeeze it to
    # make it a scalar; squeeze along axis = None since there is no column axix
    average_cross_entropy_loss = np.squeeze(total_cross_entropy_loss / n_samples, axis=None)

    # cross_entropy_loss = -np.sum(y_true * np.log(y_pred)) / n_samples
    # print(np.isclose(average_cross_entropy_loss, cross_entropy_loss))
    return average_cross_entropy_loss


def l1_loss():
    pass


class l2_loss:
    """L2 Loss (total l2 loss, to get mean l2_loss, please divide by the number of samples)"""

    # def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray = None) -> None:
    #     self.y_true = y_true
    #     self.y_pred = y_pred
    #     self.X = X

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray = None):
        """Implements L2 Loss with $l2_loss(y_true, y_pred) = \sum_{i=1}^{m} (y_true-y_pred)^2$
        Args:
            y_true (np.ndarray): [description]
            y_pred (np.ndarray): [description]
        Returns:
            [type]: [description]
        """
        l2_loss = np.sum(np.square(y_true - y_pred))
        return l2_loss

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray = None):
        """
        Compute the gradient of the l2 loss function.
        """

        gradient_vector: np.ndarray = -np.matmul((y_true - y_pred).T, X)
        # rename it to reflect its gradient of beta
        dl2_dB = gradient_vector[:]
        return dl2_dB
