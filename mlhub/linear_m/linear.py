import logging
import random
import numpy as np
import pandas as pd
from types import FunctionType

from mlhub.regularization import Regularization
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
    - reg: (str): Regularization type.
        Default is None (i.e., no regularization)
    - l1_coef (float): Regularization strength
        Default is 0
    - l2_coef (float): Regularization strength
        Default is 0
    - sgd_sample (float): Number of samples that will be used at each training iteration. Can accept either whole numbers or fractions from 0.0 to 1.0.
        Default is None.  (i.e., all samples are used).
    - random_state (float): Any number for reproducibility of the result
        Default is 42
    """

    def __init__(
        self,
        n_iter: int = 100,
        learning_rate: float = 0.1,
        weights: np.array = None,
        metric: str = None,
        reg: str = None,
        l1_coef: float = 0,
        l2_coef: float = 0,
        sgd_sample: float = None,
        random_state: int = 42,
    ) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        self._best_score = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def __str__(self) -> str:
        return f"MyLineReg class: n_iter: {self.n_iter}, learning_rate: {self.learning_rate}"

    def _add_base(self, X):
        """Append a column of ones to the feature matrix"""
        X.insert(0, "base_0", 1)

    def _initialize_weights(self, size):
        """
        Initialize weights if they are not already defined.

        Parameters:
            size (int): The size of the weights to initialize.
        """

        if self.weights is None:
            weight_size = size
            self.weights = np.ones(weight_size)
        else:
            Warning("Defined weights used")

    def _compute_gradient(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss function with respect to the weights.

        Parameters:
            X (np.ndarray): The input data.
            y_true (np.ndarray): The target labels.
            y_pred (np.ndarray): The predicted labels.

        Returns:
            np.ndarray: The computed gradient.
        """
        # Compute the gradient
        gradient = 2 * np.dot((y_pred - y_true), X) / X.shape[0]

        if self.reg in ["l1", "elastic", "elasticnet"]:
            # Regularization term for l1
            gradient += Regularization(alpha=self.l1_coef).l1(self.weights)

        if self.reg in ["l2", "elastic", "elasticnet"]:
            # Regularization term for l2
            gradient += Regularization(alpha=self.l2_coef).l2(self.weights)

        if self.reg is not None and self.reg not in ["l1", "l2", "elastic", "elasticnet"]:
            ValueError(
                "Invalid regularization type. Please choose from: 'l1', 'l2', 'elastic', or 'elasticnet'"
            )

        return gradient

    def _update_weights(self, gradient: np.ndarray, i: int) -> None:
        """
        Update weights using gradient descent.

        Parameters:
            gradient (np.ndarray): The gradient computed for the current iteration.
            i (int): The current iteration index.

        Returns:
            None
        """
        # Update weights using gradient descent
        if isinstance(self.learning_rate, (int, float)):
            # If learning rate is a constant
            self.weights -= self.learning_rate * gradient
        elif isinstance(self.learning_rate, FunctionType):
            # If learning rate is a function of the iteration index
            self.weights -= self.learning_rate(i + 1) * gradient
        else:
            raise TypeError("Learning rate must be either a constant (integer or float) or a function.")

    def _print_progress(
        self, i: int, X: np.ndarray, y: np.ndarray, mse_error: np.ndarray, verbose: int = False
    ) -> None:
        """
        Print training progress including loss and optional metric.

        Parameters:
            i (int): The current iteration index.
            X (np.ndarray): The input data.
            y (np.ndarray): The target labels.
            mse_error (np.ndarray): The mean squared error.
            verbose (int, optional): If provided, specifies the frequency of printing progress. Defaults to 1.

        """
        metric_text = ""

        if verbose and i % verbose == 0:
            if self.metric is not None:
                # If metric is specified, calculate and add metric text
                metric_text = f"| {self.metric}: {RegressionMetric._calculate_metric(metric_name=self.metric, y_true=y, y_predict=self._make_prediction(X, self.weights))}"

            # Print progress information
            self.logger.info(f"{i} | loss: {np.mean(mse_error ** 2)} {metric_text}")

    def _make_prediction(self, X: np.array, W: np.array) -> None:
        """
        Make predictions using the calculated weights.
        Parameters:
        - X (array-like): Features as a pandas DataFrame.
        - W (array-like): Model weights.

        Returns:
        - array-like: Predicted target variable values.
        """
        return np.dot(X, W)

    def _add_best_score(self, X: np.ndarray, y_true: np.ndarray) -> None:
        """
        Update the best score if a metric is specified.

        Parameters:
            X (np.ndarray): The input data.
            y_true (np.ndarray): The true target labels.
        """
        if self.metric is not None:
            self._best_score = RegressionMetric._calculate_metric(
                metric_name=self.metric,
                y_true=y_true,
                y_predict=self._make_prediction(X=X, W=self.weights),
            )

    def _get_batch(self, X: pd.DataFrame, y_true: pd.Series):
        """
        Form a mini-batch of data.

        Args:
            X (DataFrame or array-like): Input features.
            y_true (Series or array-like): True labels.

        Returns:
            X_batch (DataFrame or array-like): Mini-batch of input features.
            y_batch (Series or array-like): Mini-batch of true labels.
        """
        # Validate sample size
        if self.sgd_sample is not None:
            if isinstance(self.sgd_sample, int):
                sample_size = min(self.sgd_sample, X.shape[0])

            elif isinstance(self.sgd_sample, float) and self.sgd_sample <= 1:
                sample_size = min(int(self.sgd_sample * X.shape[0]), X.shape[0])
            else:
                raise ValueError("Invalid type for sgd_sample. Must be int or float.")

            # Create new baths
            sample_rows_idx = random.sample(range(X.shape[0]), sample_size)
            X_batch = X.iloc[sample_rows_idx]
            y_batch = y_true.iloc[sample_rows_idx]

        else:
            X_batch = X
            y_batch = y_true

        return X_batch, y_batch

    def fit(self, X: pd.DataFrame, y_true: pd.Series, verbose: int = False) -> None:
        """
        Train linear regression on the given data.

        Parameters:
        - X (DataFrame): Features as a pandas DataFrame.
        - y (Series): Target variable as a pandas Series.
        - verbose (int or bool): Indicates at which iteration to print logs. Default is False (i.e., nothing is printed).
        """
        random.seed(self.random_state)
        self._add_base(X=X)
        self._initialize_weights(size=X.shape[1])

        # Gradient descent iterations
        for i in range(self.n_iter):
            X_batch, y_batch = self._get_batch(X=X, y_true=y_true)
            y_pred = self._make_prediction(X_batch, self.weights)

            # Calculate Mean Squared Error (MSE)
            mse_error = np.mean((y_pred - y_batch) ** 2)

            # Compute the gradient
            gradient = self._compute_gradient(X=X_batch, y_true=y_batch, y_pred=y_pred)

            # Update weights using gradient descent
            self._update_weights(gradient=gradient, i=i)

            self._print_progress(i=i, X=X_batch, y=y_batch, mse_error=mse_error, verbose=verbose)

            self._add_best_score(X=X_batch, y_true=y_batch)

    def get_coef(self):
        """
        Return model coefficients, starting from the second value.

        Returns:
        - array-like: Model coefficients starting from the second value.
        """
        if self.weights is None:
            raise ValueError("Weights are not initialized.")
        return self.weights[1:]

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
