import logging
import numpy as np
import pandas as pd

from mlhub.metrics.regression_metrics import RegressionMetric


class MyLineReg:
    """
    Class for training linear regression using gradient descent.

    Parameters:
    - learning_rate (float): Learning rate for gradient descent
        Default is 0.1.
    - n_iter (int): Number of iterations for gradient descent
        Default is 100.
    - weights (array-like): Model weights.
        Default is None (i.e., all weights are set to 1).
    - metric (str): Metric for evaluation during training. Options: 'mae', 'mse', 'rmse', 'mape', 'r2'.
        Default is None (i.e., no metric is calculated).
    """

    def __init__(
        self,
        n_iter: int = 100,
        learning_rate: float = 0.1,
        weights: np.array = None,
        metric: str = None,
    ) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self._best_score = None
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

            metric_text = ""
            if verbose and i % verbose == 0:
                if self.metric is not None:
                    metric_text = f"| {self.metric}: {RegressionMetric._calculate_metric(metric_name=self.metric, y_true=y, y_predict=self._make_prediction(X, self.weights))}"
                self.logger.info(f"{i} | loss: {np.mean(mse_error ** 2)} {metric_text}")

        if self.metric is not None:
            self._best_score = RegressionMetric._calculate_metric(
                metric_name=self.metric, y_true=y, y_predict=self._make_prediction(X, self.weights)
            )

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
        if self.weights is None:
            raise ValueError("Weights are not initialized. Please train the model first.")
        if X.shape[1] < self.weights.shape[1]:
            self._add_base(X)
        return self._make_prediction(X.values, self.weights)

    def get_best_score(self):
        """
        Return the last value of the metric after the model has been trained.
        Returns:
        - float: Last value of the metric.
        """
        return self._best_score
