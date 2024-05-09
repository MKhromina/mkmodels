import pandas as pd
import numpy as np
from mlhub.knn.knn_base import KNNBase


class MyKNNClf(KNNBase):
    """
    Custom k-Nearest Neighbors classifier.
    """

    def __init__(self, k: int = 3, metric: str = "euclidean", weight: str = "uniform"):
        """
        Initialize the MyKNNClf classifier.

        Parameters:
            - k (int, optional): Number of neighbors to consider. Defaults to 3.
            - metric (str, optional): Name of the distance metric to use. Defaults to None.
        """
        super().__init__(k=k, metric=metric, weight=weight)

    def _get_weighted_knn(self, distance: np.ndarray) -> np.ndarray:
        """
        Applies the weighted k-nearest neighbors algorithm to calculate predictions.

        Parameters:
            - distance (np.ndarray): The distance array computed by the k-nearest neighbors algorithm.

        Returns:
            np.ndarray: The predictions based on the weighted k-nearest neighbors algorithm.
        """
        neighbors_label = self.y[np.argsort(distance)][:, : self.k]
        neighbors_distance = np.sort(distance)[:, : self.k]

        if self.weight == "uniform":
            return np.mean(neighbors_label, axis=1)

        elif self.weight == "rank":
            ranks = 1 / np.arange(1, self.k + 1)
            class_ranks = np.where(neighbors_label == 1, ranks, 0)
            return np.sum(class_ranks / np.sum(ranks), axis=1)

        elif self.weight == "distance":
            class_distance = 1 / np.where(neighbors_label == 1, neighbors_distance, np.inf)
            total_distance = np.sum(1 / neighbors_distance, axis=1)
            return np.sum(class_distance / total_distance[:, np.newaxis], axis=1)

    def predict(self, X_test: pd.DataFrame, threshold=0.5) -> np.ndarray:
        """
        Predict labels for test data using the k-Nearest Neighbors algorithm.

        Parameters:
            - X_test (pd.DataFrame): Test data.
        Returns:
            np.ndarray: Predicted labels.
        """
        y_pred = self.predict_proba(X_test)
        return (y_pred >= threshold).astype(int)

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities for each class for test data using the k-Nearest Neighbors algorithm.

        Parameters:
            - X_test (pd.DataFrame): Test data.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        return super().predict(X_test=X_test)
