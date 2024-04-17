import numpy as np
import pandas as pd
from mlhub.linear_m.base_linear import BaseRegression
from mlhub.metrics.classification_metrics import ClassificationMetric


class MyLogReg(BaseRegression):
    """
    Class for training logistic regression using gradient descent.

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

    def _make_prediction(self, X: pd.DataFrame, W: np.array):
        """
        Make predictions using logistic regression.

        Parameters:
            X (pd.DataFrame): Input features.
            W (np.array): Model weights.

        Returns:
            np.array: Predicted probabilities.
        """
        # Calculate the linear combination of features and weights
        z = super()._make_prediction(X=X, W=self.weights)
        # Apply the logistic function to the linear combination
        return 1 / (1 + np.exp(-z))

    def _compute_error(self, y_pred: pd.Series, y_batch: pd.Series, eps=1e-15):
        """
        Compute the logistic loss between predicted and actual values.

        Parameters:
            y_pred (pd.Series): Predicted probabilities.
            y_batch (pd.Series): Actual target values.
            eps (float): Small value to avoid numerical instability.

        Returns:
            float: Logistic loss.
        """
        # Calculate the logistic loss using the cross-entropy formula
        return -np.mean(y_batch * np.log(y_pred + eps) + (1 - y_batch) * np.log(y_pred + eps))

    def _get_metric(self, y_true: pd.Series, y_predict: pd.Series) -> str:
        """
        Get the value of the specified metric.

        Parameters:
        - X (DataFrame): Features as a pandas DataFrame.
        - y_true (Series): True target labels.

        Returns:
        - str: Text representation of the metric value.
        """
        metric_instance = ClassificationMetric()
        metric_value = metric_instance._calculate_metric(
            metric_name=self.metric, y_true=y_true, y_predict=y_predict
        )
        metric_text = f"| {self.metric}: {metric_value}"
        return metric_text

    def _add_best_score(self, X: np.ndarray, y_true: np.ndarray) -> None:
        """
        Update the best score if a metric is specified.

        Parameters:
            X (np.ndarray): The input data.
            y_true (np.ndarray): The true target labels.
        """
        if isinstance(self.metric, str):
            metric_instance = ClassificationMetric()
            if self.metric != "roc_auc":
                self._best_score = metric_instance._calculate_metric(
                    metric_name=self.metric,
                    y_true=y_true,
                    y_predict=self.predict(X=X),
                )
            elif self.metric == "roc_auc":
                self._best_score = metric_instance._calculate_metric(
                    metric_name=self.metric,
                    y_true=y_true,
                    y_predict=self.predict_proba(X=X),
                )
        elif self.metric is None:
            pass
        else:
            raise ValueError("Invalid metric type. Metric must be a string.")

    def predict_proba(self, X):
        """
        Predicts class probabilities for the input data.

        Parameters:
        - X (array-like): Input data.

        Returns:
        - array-like: Probabilities of belonging to classes.
        """
        return super().predict(X=X)

    def predict(self, X, trashed=0.5):
        """
        Predicts class labels for the input data.

        Parameters:
        - X (array-like): Input data.
        - threshold (float, optional): Probability threshold for classification. Default is 0.5.

        Returns:
        - array-like: Class labels.
        """
        res = self.predict_proba(X=X)
        return res > trashed
