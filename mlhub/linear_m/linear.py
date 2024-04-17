import numpy as np
import pandas as pd

from mlhub.linear_m.base_linear import BaseRegression
from mlhub.metrics.regression_metrics import RegressionMetric


class MyLineReg(BaseRegression):
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
        super().__init__(
            n_iter=n_iter,
            learning_rate=learning_rate,
            weights=weights,
            metric=metric,
            reg=reg,
            l1_coef=l1_coef,
            l2_coef=l2_coef,
            sgd_sample=sgd_sample,
            random_state=random_state,
        )

    def _compute_error(self, y_pred: pd.Series, y_batch: pd.Series):
        """
        Calculate the mean squared error between predicted and actual values.

        Parameters:
            y_pred (pd.Series): Predicted values.
            y_batch (pd.Series): Actual values.

        Returns:
            float: Mean squared error.
        """
        return np.mean((y_pred - y_batch) ** 2)

    def _get_gradient(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of the loss function with respect to the weights.

        Parameters:
            X (np.ndarray): Feature matrix.
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.

        Returns:
            np.ndarray: Gradient of the loss function with respect to the weights.
        """
        gradient = super()._get_gradient(X=X, y_true=y_true, y_pred=y_pred)
        # Multiply the gradient by 2 for linear regression
        gradient *= 2
        return gradient

    def _get_metric(self, y_true: pd.Series, y_predict: pd.Series) -> str:
        """
        Get the value of the specified metric.

        Parameters:
        - X (DataFrame): Features as a pandas DataFrame.
        - y_true (Series): True target labels.

        Returns:
        - str: Text representation of the metric value.
        """
        metric_instance = RegressionMetric()
        metric = metric_instance._calculate_metric(
            metric_name=self.metric, y_true=y_true, y_predict=y_predict
        )
        metric_text = f"| {self.metric}: {metric}"
        return metric_text

    def _add_best_score(self, X: np.ndarray, y_true: np.ndarray) -> None:
        """
        Update the best score if a metric is specified.

        Parameters:
            X (np.ndarray): The input data.
            y_true (np.ndarray): The true target labels.
        """
        if self.metric is not None:
            metric_instance = RegressionMetric()
            self._best_score = metric_instance._calculate_metric(
                metric_name=self.metric,
                y_true=y_true,
                y_predict=self._make_prediction(X=X, W=self.weights),
            )
