import numpy as np


class Regularization:
    """
    Class for implementing regularization methods.

    Parameters:
        - alpha (float): Regularization strength (default is 1.0).
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def l1(self, weights: np.ndarray):
        """
        Calculate L1 regularization.

        Parameters:
            - weights (numpy.ndarray): Model weights.

        Returns:
            float: L1 regularization penalty.
        """
        return self.alpha * np.sign(weights)

    def l2(self, weights: np.ndarray):
        """
        Calculate L2 regularization.

        Parameters:
            weights (numpy.ndarray): Model weights.

        Returns:
            float: L2 regularization penalty.
        """
        return self.alpha * 2 * weights
