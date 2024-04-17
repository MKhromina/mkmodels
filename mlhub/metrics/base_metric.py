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
            return func(y_true, y_predict)

        return wrapper

    @staticmethod
    def _convert_dtype(y: Sequence) -> pd.Series:
        """
        Convert input data to pandas Series if it's not already.

        Parameters:
        - y (array-like): Input data.

        Returns:
        - pd.Series: Data converted to pandas Series.
        """
        if isinstance(y, pd.Series):
            return y
        elif isinstance(y, (np.ndarray, list, tuple)):
            return pd.Series(y)
        else:
            raise TypeError("Input data must be either numpy array or pandas Series.")

    def _calculate_metric(self, metric_name: str, y_true: Sequence, y_predict: Sequence, **kwargs):
        """
        Calculate the specified metric.

        Parameters:
        - metric_name: Name of the metric to calculate.
        - y_true: True labels.
        - y_predict: Predicted labels.

        Returns:
        - float: Calculated metric.
        """
        # Get a metric by key
        if metric_name not in self._metrics_dict:
            raise ValueError(
                f"Unknown metric: {metric_name}, try one of {', '.join(self._metrics_dict.keys())}"
            )

        # Call the corresponding function to calculate the metric
        metric_func = self._metrics_dict[metric_name]
        return metric_func(y_true=y_true, y_predict=y_predict, **kwargs)
