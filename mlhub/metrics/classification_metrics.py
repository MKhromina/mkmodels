import numpy as np
import pandas as pd
from types import MappingProxyType
from typing import Union, Sequence
from mlhub.metrics.base_metric import BaseMetric


class ClassificationMetric(BaseMetric):
    """
    Class for computing classification metrics (accuracy, recall, precision, fb and f1 score, roc_auc)
    """

    def __init__(self) -> None:
        """
        Initialize the ClassificationMetric class.
        """
        # Create a dictionary where keys are metric names and values are corresponding methods
        self._metrics_dict = MappingProxyType(
            {
                "accuracy": self.accuracy,
                "recall": self.recall,
                "precision": self.precision,
                "f1": self.f_score,
                "roc_auc": self.roc_auc,
            }
        )

    @staticmethod
    @BaseMetric.check_length
    def _get_cm(y_true: Sequence, y_predict: Sequence) -> pd.DataFrame:
        """
        Get the confusion matrix.

        Parameters:
        - y_true: True labels.
        - y_predict: Predicted labels.

        Returns:
            pd.DataFrame: Confusion matrix.
        """
        y_true = BaseMetric._convert_dtype(y_true)
        y_predict = BaseMetric._convert_dtype(y_predict)
        confusion_matrix = pd.crosstab(y_true, y_predict)
        return confusion_matrix

    @staticmethod
    @BaseMetric.check_length
    def accuracy(y_true: Sequence, y_predict: Sequence) -> float:
        """
        Compute the accuracy metric.

        Parameters:
        - y_true: True labels.
        - y_predict: Predicted labels.

        Returns:
            float: Accuracy score.
        """

        confusion_matrix = ClassificationMetric._get_cm(y_true=y_true, y_predict=y_predict)
        correct_predictions = np.diag(confusion_matrix).sum()
        total_predictions = confusion_matrix.to_numpy().sum()

        if total_predictions == 0:
            raise ValueError("Total predictions cannot be zero.")

        accuracy = correct_predictions / total_predictions
        return accuracy

    # TODO: think about multi class

    @staticmethod
    @BaseMetric.check_length
    def precision(
        y_true: Union[pd.Series, np.ndarray], y_predict: Union[pd.Series, np.ndarray], positive_label=1
    ) -> float:
        """
        Compute the precision metric.

        Parameters:
        - y_true: True labels.
        - y_predict: Predicted labels.
        - positive_label: Value representing the positive class. Default is 1.

        Returns:
            float: Precision score.
        """
        confusion_matrix = ClassificationMetric._get_cm(y_true=y_true, y_predict=y_predict)

        true_positive = confusion_matrix[positive_label][positive_label]
        predict_positive = confusion_matrix[positive_label].sum()

        return true_positive / predict_positive if predict_positive != 0 else 0

    @staticmethod
    @BaseMetric.check_length
    def recall(
        y_true: Union[pd.Series, np.ndarray], y_predict: Union[pd.Series, np.ndarray], positive_label=1
    ) -> float:
        """
        Compute the recall metric.

        Parameters:
        - y_true: True labels.
        - y_predict: Predicted labels.
        - positive_label: Value representing the positive class. Default is 1.

        Returns:
            float: Recall score.
        """
        confusion_matrix = ClassificationMetric._get_cm(y_true=y_true, y_predict=y_predict)

        true_positive = confusion_matrix[positive_label][positive_label]
        all_positive = confusion_matrix.loc[positive_label].sum()
        return true_positive / all_positive if all_positive != 0 else 0

    @staticmethod
    @BaseMetric.check_length
    def f_score(
        y_true: Union[pd.Series, np.ndarray],
        y_predict: Union[pd.Series, np.ndarray],
        b: float = 1,
        positive_label=1,
    ) -> float:
        """
        Compute the F-score metric.

        Parameters:
        - y_true: True labels.
        - y_predict: Predicted labels.
        - b: Beta parameter for balancing precision and recall. Default is 1.
        - positive_label: Value representing the positive class. Default is 1.

        Returns:
            float: F-score.
        """
        precision = ClassificationMetric.precision(
            y_true=y_true, y_predict=y_predict, positive_label=positive_label
        )
        recall = ClassificationMetric.recall(
            y_true=y_true, y_predict=y_predict, positive_label=positive_label
        )
        return (
            (1 + b**2) * (precision * recall) / ((b**2 * precision) + recall)
            if (precision + recall) != 0
            else 0
        )

    @staticmethod
    @BaseMetric.check_length
    def roc_auc(
        y_true: Union[pd.Series, np.ndarray], y_predict: Union[pd.Series, np.ndarray], b=1
    ) -> float:
        """
        Compute the ROC AUC score for binary classification using pandas DataFrames.

        Parameters:
        - y_true: pd.Series containing true binary labels (0 or 1).
        - y_predict: pd.Series containing predicted scores (probability estimates).

        Returns:
            float: ROC AUC score.
        """
        roc_auc_df = pd.DataFrame({"true": y_true, "pred": y_predict}).sort_values(
            by="pred", ascending=False
        )

        ## Calculate 'positive' score
        roc_auc_df["shift"] = roc_auc_df["true"].shift(fill_value=0).cumsum()

        mask = roc_auc_df.groupby("pred")["shift"].agg(["min", "max"]).diff(axis=1)["max"] / 2
        roc_auc_df["mask"] = roc_auc_df["pred"].map(mask)
        roc_auc_df.loc[roc_auc_df["true"].astype(bool), ["shift", "mask"]] = 0
        roc_auc_df["shift"] -= roc_auc_df["mask"]

        # Calculate ROC AUC
        res = roc_auc_df["shift"].sum() / roc_auc_df["true"].value_counts().prod()
        return res
