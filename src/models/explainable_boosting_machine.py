import mlflow.pyfunc
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


class EBM(BaseEstimator, ClassifierMixin, mlflow.pyfunc.PythonModel):
    def __init__(self, random_state=42):
        super().__init__()
        self.random_state = random_state
        self.ebm_model = ExplainableBoostingClassifier(random_state=self.random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EBM":
        """EBM fit method.

        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Target.

        Returns:
            EBM: EBM fitted model / class.
        """
        self.ebm_model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """_summary_

        Args:
            X (pd.DataFrame): Features.

        Returns:
            np.ndarray: Model predictions.
        """
        model_predictions = self.ebm_model.predict(X)
        return model_predictions
