import numpy as np
from types import MappingProxyType


class RegressionMetric:
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
    def mse(y_true: np.array, y_predict: np.array):
        """
        Compute the mean squared error (MSE) between true and predicted values.

        Parameters:
        - y_true (array-like): True target values.
        - y_pred (array-like): Predicted target values.

        Returns:
        - float: Mean squared error.
        """
        return np.mean((y_true - y_predict) ** 2)

    @staticmethod
    def mae(y_true: np.array, y_predict: np.array):
        """
        Compute the mean absolute error (MAE) between true and predicted values.

        Parameters:
        - y_true (array-like): True target values.
        - y_pred (array-like): Predicted target values.

        Returns:
        - float: Mean absolute error.
        """
        return np.mean(np.abs(y_true - y_predict))

    @staticmethod
    def rmse(y_true: np.array, y_predict: np.array):
        """
        Root Mean Squared Error (RMSE) between true and predicted values.

        Parameters:
        - y_true (array-like): True target values.
        - y_pred (array-like): Predicted target values.

        Returns:
        - float: Mean squared error.
        """
        return np.power(np.mean(np.power((y_true - y_predict), 2)), 0.5)

    @staticmethod
    def r2(y_true: np.array, y_predict: np.array):
        """
        Coefficient of determination (R^2) between true and predicted values.

        Parameters:
        - y_true (array-like): True target values.
        - y_pred (array-like): Predicted target values.

        Returns:
        - float: coefficient of determination.
        """
        return 1 - (
            sum(np.power((y_true - y_predict), 2)) / sum(np.power((y_true - np.mean(y_true)), 2))
        )

    @staticmethod
    def mape(y_true: np.array, y_predict: np.array):
        """
        Mean Absolute Percentage Error (MAPE) between true and predicted values.

        Parameters:
        - y_true (array-like): True target values.
        - y_pred (array-like): Predicted target values.

        Returns:
        - float: Mean Absolute Percentage Error.
        """
        return 100 * np.mean(np.abs((y_true - y_predict) / y_true))

    @staticmethod
    def _calculate_metric(metric_name: str, y_true: np.array, y_predict: np.array):
        # Get a metric by key
        if metric_name not in RegressionMetric()._metrics_dict:
            raise ValueError(f"Unknown metric: {metric_name}")

        # Call the corresponding function to calculate the metric
        metric_func = RegressionMetric()._metrics_dict[metric_name]
        return metric_func(y_true=y_true, y_predict=y_predict)


bl
