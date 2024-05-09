import pandas as pd
import numpy as np
from typing import Union, Sequence
from abc import ABC, abstractmethod


class BaseMetric(ABC):

    """
    Base class for calculating metrics in machine learning tasks.
    """

    def __init__(self) -> None:
        """
        Initializes the BaseMetric class.
        """
        self._metrics_dict = {}

    @staticmethod
    def check_length(func):
        """
        Decorator function to check the length of true and predicted labels.
        """

        def wrapper(y_true: Sequence, y_predict: Sequence, *args, **kwargs):
            """
            Wrapper function to check length of true and predicted labels.
            """
            if len(y_true) != len(y_predict):
                raise ValueError("Lengths of true labels and predicted labels must be equal.")
            return func(y_true, y_predict, *args, **kwargs)

        return wrapper
    
    @staticmethod
    def check_shape(func):
        """
        Wrapper function to check second dimension shape of input arrays.
        """

        def wrapper(X_train: np.ndarray, X_test: np.ndarray, *args, **kwargs):
            """
            Wrapper function to check length of true and predicted labels.
            """
            if X_train.shape[1] != X_test.shape[1]:
                raise ValueError("Second dimension of input arrays must be equal.")
            return func(X_train, X_test, *args, **kwargs)

        return wrapper

    @staticmethod
    def _convert_dtype_to_series(y: Sequence) -> pd.Series:
        """
        Convert input data to pandas Series if it's not already.

        Parameters:
            - y (array-like): Input data.

        Returns:
            pd.Series: Data converted to pandas Series.
        """
        if isinstance(y, pd.Series):
            return y
        elif isinstance(y, (np.ndarray, list, tuple)):
            return pd.Series(y)
        else:
            raise TypeError("Input data must be either numpy array or pandas Series.")
        
    @staticmethod
    def _convert_dtype_to_numpy(x: pd.DataFrame) -> np.ndarray:
        """
        Convert input data to pandas Series if it's not already.

        Parameters:
            - x (pd.DataFramr): Input data.

        Returns:
            np.ndarray: Data converted to numpy array.
        """
        if isinstance(x, np.ndarray):
            return x
        elif isinstance(x, (pd.DataFrame, pd.Series)):
            return x.to_numpy()
        else:
            raise TypeError("Input data must be pd.DataFrame or pd.Series.")

    def _calculate_metric(self, metric_name: str, true_value: Sequence, predict_value: Sequence, **kwargs):
        """
        Calculate the specified metric.

        Parameters:
            - metric_name: Name of the metric to calculate.
            - true_value: True labels.
            - predict_value: Predicted labels.

        Returns:
            float: Calculated metric.
        """
        # Get a metric by key
        if metric_name not in self._metrics_dict:
            raise ValueError(
                f"Unknown metric: {metric_name}, try one of {', '.join(self._metrics_dict.keys())}"
            )

        # Call the corresponding function to calculate the metric
        metric_func = self._metrics_dict[metric_name]
        return metric_func(true_value, predict_value, **kwargs)
