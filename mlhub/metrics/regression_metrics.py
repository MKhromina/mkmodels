import numpy as np
from types import MappingProxyType
from mlhub.metrics.base_metric import BaseMetric


class RegressionMetric(BaseMetric):
    def __init__(self) -> None:
        # Create a dictionary where keys are metric names and values are corresponding methods
        self._metrics_dict = MappingProxyType(
            {
                "mse": RegressionMetric.mse,
                "mae": RegressionMetric.mae,
                "rmse": RegressionMetric.rmse,
                "r2": RegressionMetric.r2,
                "mape": RegressionMetric.mape,
            }
        )

    @staticmethod
    @BaseMetric.check_length
    def mse(y_true: np.ndarray, y_predict: np.ndarray) -> float:
        """
        Compute the mean squared error (MSE) between true and predicted values.

        Parameters:
            - y_true (array-like): True target values.
            - y_pred (array-like): Predicted target values.

        Returns:
            float: Mean squared error.
        """
        return np.mean((y_true - y_predict) ** 2)

    @staticmethod
    @BaseMetric.check_length
    def mae(y_true: np.ndarray, y_predict: np.ndarray) -> float:
        """
        Compute the mean absolute error (MAE) between true and predicted values.

        Parameters:
            - y_true (array-like): True target values.
            - y_pred (array-like): Predicted target values.

        Returns:
            float: Mean absolute error.
        """
        return np.mean(np.abs(y_true - y_predict))

    @staticmethod
    @BaseMetric.check_length
    def rmse(y_true: np.ndarray, y_predict: np.ndarray) -> float:
        """
        Root Mean Squared Error (RMSE) between true and predicted values.

        Parameters:
            - y_true (array-like): True target values.
            - y_pred (array-like): Predicted target values.

        Returns:
            float: Mean squared error.
        """
        return np.power(np.mean(np.power((y_true - y_predict), 2)), 0.5)

    @staticmethod
    @BaseMetric.check_length
    def r2(y_true: np.ndarray, y_predict: np.ndarray) -> float:
        """
        Coefficient of determination (R^2) between true and predicted values.

        Parameters:
            - y_true (array-like): True target values.
            - y_pred (array-like): Predicted target values.

        Returns:
            float: coefficient of determination.
        """
        return 1 - (
            sum(np.power((y_true - y_predict), 2)) / sum(np.power((y_true - np.mean(y_true)), 2))
        )

    @staticmethod
    @BaseMetric.check_length
    def mape(y_true: np.ndarray, y_predict: np.ndarray) -> float:
        """
        Mean Absolute Percentage Error (MAPE) between true and predicted values.

        Parameters:
            - y_true (array-like): True target values.
            - y_pred (array-like): Predicted target values.

        Returns:
            float: Mean Absolute Percentage Error.
        """
        return 100 * np.mean(np.abs((y_true - y_predict) / y_true))
