from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


class EBM(BaseEstimator, ClassifierMixin):
    def __init__(self):
        super().__init__()
        self.ebm_model = None

    def fit(self, X, y):
        self.ebm_model = ExplainableBoostingClassifier(random_state=42)
        self.ebm_model.fit(X, y)
        return self

    def predict(self, X):
        model_predictions = self.ebm_model.predict(X)
        return model_predictions
