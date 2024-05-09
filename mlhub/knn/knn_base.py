import numpy as np
import pandas as pd
from types import MappingProxyType
from typing import Union, Sequence
from abc import ABC, abstractmethod

from mlhub.metrics.distance_metrics import DistanceMetric


class KNNBase(ABC):
    """
    Abstract class for k Nearest Neighbors algorithm
    """

    def __init__(self, k: int, metric: str = "euclidean", weight: str = None):
        """
        Initializes the BaseLinear class with the specified number of neighbors (k).

        Parameters:
            - k (int): The number of neighbors to consider in the k Nearest Neighbors algorithm.
            - metric (str, optional): The distance metric to use.
                Default is "euclidean".
            - weight (str or None)
            Default is None
        """
        self.k = k
        self.metric = metric
        self.weight = weight
        self.train_size = None
        self.X = None
        self.y = None

    def __repr__(self) -> str:
        """
        Returns a string representation of the KNNBase object.

        Returns:
            str: String representation of the KNNBase object.
        """
        return f"{self.__class__.__name__} class: k={self.k}"

    def _convert_to_array(self, obj: np.ndarray) -> np.ndarray:
        if not isinstance(obj, np.ndarray):
            obj = np.asarray(obj)
        return obj

    def fit(self, X: pd.DataFrame, y_true: pd.Series) -> None:
        """
        Fit the KNNBase model to the training data.

        Parameters:
            - X (pd.DataFrame): Training features.
            - y_true (pd.Series): True labels.
        """
        self.train_size = X.shape
        self.X = self._convert_to_array(X)
        self.y = self._convert_to_array(y_true)

    def _validate(self):
        """
        Validates the model by checking if it has been trained and if the parameters are valid.
        """
        if self.X is None or self.y is None:
            raise ValueError("Model should be trained first")
        if self.k is None or not isinstance(self.k, int) or self.k <= 0:
            raise ValueError("Parameter k should be a positive integer value")

    @abstractmethod
    def _get_weighted_knn(self, distance: np.ndarray) -> np.ndarray:
        """
        Applies the weighted k-nearest neighbors algorithm to calculate predictions.
        """
        pass

    @abstractmethod
    def predict(self, X_test: Union[np.ndarray, pd.DataFrame]):
        """
        Abstract method to predict labels for test data.

        Parameters:
            - X_test: (array like) Test data.
        """
        self._validate()
        metric_instance = DistanceMetric()
        X_test = self._convert_to_array(X_test)
        distance = metric_instance._calculate_metric(
            metric_name=self.metric, true_value=self.X, predict_value=X_test
        )
        return self._get_weighted_knn(distance=distance)
