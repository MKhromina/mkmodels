import numpy as np
from types import MappingProxyType
from mlhub.metrics.base_metric import BaseMetric
from mlhub.metrics.base_metric import BaseMetric


class DistanceMetric(BaseMetric):
    def __init__(self) -> None:
        """
        Initialize DistanceMetric class.
        """
        self._metrics_dict = MappingProxyType(
            {
                "euclidean": self.euclidean,
                "manhattan": self.manhattan,
                "chebyshev": self.chebyshev,
                "cosine": self.cosine,
            }
        )

    @staticmethod
    @BaseMetric.check_shape
    def euclidean(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        """
        Calculate the Euclidean distance between X_train and X_test.

        Parameters:
            - X_train (np.ndarray): Train features.
            - X_test (np.ndarray): Test features.

        Returns:
            np.ndarray: Euclidean distances.
        """
        return np.sqrt(np.sum(np.abs(X_test[:, np.newaxis] - X_train) ** 2, axis=2))

    @staticmethod
    @BaseMetric.check_shape
    def manhattan(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        """
        Calculate the Manhattan distance between X_train and X_test.

        Parameters:
            - X_train (np.ndarray): Train features.
            - X_test (np.ndarray): Test features.

        Returns:
            np.ndarray: Manhattan distances.
        """
        return np.sum(np.abs(X_test[:, np.newaxis] - X_train), axis=2)

    @staticmethod
    @BaseMetric.check_shape
    def chebyshev(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        """
        Calculate the Chebyshev distance between X_train and X_test_

        Parameters:
            - X_train (np.ndarray): Train features.
            - X_test (np.ndarray): Test features.

        Returns:
            np.ndarray: Chebyshev distances.
        """
        return np.max(np.abs(X_test[:, np.newaxis] - X_train), axis=2)

    @staticmethod
    @BaseMetric.check_shape
    def cosine(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        """
        Calculate the cosine similarity between X_train and X_test.

        Parameters:
            - X_train (np.ndarray): Train features.
            - X_test (np.ndarray): Test features.

        Returns:
            np.ndarray: Cosine similarities.
        """
        x_train_norm = np.linalg.norm(X_train, axis=1)
        x_test_norm = np.linalg.norm(X_test, axis=1)
        similarity = np.dot(X_test, X_train.T) / (x_test_norm[:, np.newaxis] * x_train_norm)
        return 1 - similarity
