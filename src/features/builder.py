from sklearn.base import BaseEstimator, TransformerMixin
from core.contracts import FEATURE_COLUMNS

class LMSFeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, feature_columns=None):
        self.feature_columns = feature_columns or FEATURE_COLUMNS

    def fit(self, X, y=None):
        self.fill_values_ = X[self.feature_columns].median(numeric_only=True)
        return self

    def transform(self, X):
        X_out = X[self.feature_columns].copy()
        X_out = X_out.fillna(self.fill_values_)
        return X_out
