import logging
import numpy as np
import pandas as pd


class MyLineReg:
    """
    Class for training linear regression using gradient descent.

    Parameters:
    - learning_rate (float): Learning rate for gradient descent (default 0.1).
    - n_iter (int): Number of iterations for gradient descent (default 100).
    - weights (array-like): Model weights. Default is None (i.e., all weights are set to 1).
    """

    def __init__(self, n_iter: int = 100, learning_rate: float = 0.1, weights: np.array = None) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def __str__(self) -> str:
        return f"MyLineReg class: n_iter: {self.n_iter}, learning_rate: {self.learning_rate}"

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = False):
        """
        Train linear regression on the given data.

        Parameters:
        - X (DataFrame): Features as a pandas DataFrame.
        - y (Series): Target variable as a pandas Series.
        - verbose (int or bool): Indicates at which iteration to print logs. Default is False (i.e., nothing is printed).
        """
        self._add_base(X)
        # Determine the number of features and create a weight vector
        if self.weights is None:
            weight_size = X.shape[1]
            self.weights = np.ones(weight_size)
        else:
            Warning("Defined weights used")

        # Gradient descent iterations
        for i in range(self.n_iter):
            # Predict y
            y_pred = self._make_prediction(X, self.weights)

            # Calculate Mean Squared Error (MSE)
            mse_error = np.mean((y_pred - y) ** 2)

            # Compute the gradient
            gradient = 2 * np.dot((y_pred - y), X) / X.shape[0]

            # Update weights using gradient descent
            self.weights -= self.learning_rate * gradient

            if verbose and i % verbose == 0:
                self.logger.info(f"{i} | loss: {np.mean(mse_error ** 2)}")

    def _add_base(self, X):
        """Append a column of ones to the feature matrix"""
        X.insert(0, "base_0", 1)

    def _make_prediction(self, X: np.array, W: np.array):
        """
        Make predictions using the calculated weights.
        Parameters:
        - X (array-like): Features as a pandas DataFrame.
        - W (array-like): Model weights.

        Returns:
        - array-like: Predicted target variable values.
        """
        return np.dot(X, W)

    def get_coef(self):
        """
        Return model coefficients, starting from the second value.

        Returns:
        - array-like: Model coefficients starting from the second value.
        """
        if self.weights is None:
            raise ValueError("Weights are not initialized.")
        return self.weights[:1]

    def predict(self, X: pd.DataFrame):
        """
        Make predictions using the trained linear regression model.

        Parameters:
        - X (DataFrame): Features as a pandas DataFrame.

        Returns:
        - array-like: Predicted target variable values.
        """
        self._add_base(X)
        return self._make_prediction(X.values, self.weights)
