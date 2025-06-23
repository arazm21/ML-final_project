# pipeline_functions.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold

class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Remove features with correlation above a given threshold.
    """
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.to_drop_ = []

    def fit(self, X, y=None):
        corr_matrix = pd.DataFrame(X).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop_ = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return self

    def transform(self, X):
        return pd.DataFrame(X).drop(columns=self.to_drop_, errors='ignore')


class CovarianceFilter(BaseEstimator, TransformerMixin):
    """
    Remove features with very low covariance with target.
    """
    def __init__(self, threshold=1e-5):
        self.threshold = threshold
        self.keep_columns_ = []

    def fit(self, X, y):
        df = pd.DataFrame(X)
        cov = df.apply(lambda col: np.cov(col, y)[0, 1])
        self.keep_columns_ = cov[abs(cov) > self.threshold].index.tolist()
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.keep_columns_]


class ConstantFeatureFilter(BaseEstimator, TransformerMixin):
    """
    Remove features with zero variance (constant features).
    """
    def __init__(self):
        self.selector = VarianceThreshold(threshold=0.0)

    def fit(self, X, y=None):
        self.selector.fit(X)
        return self

    def transform(self, X):
        return self.selector.transform(X)


class IQRFilter(BaseEstimator, TransformerMixin):
    """
    Removes rows with any feature value outside the IQR range (Q1 - 1.5*IQR to Q3 + 1.5*IQR).
    """
    def __init__(self, multiplier=1.5):
        self.multiplier = multiplier
        self.bounds_ = {}

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        self.bounds_ = {
            col: (Q1[col] - self.multiplier * IQR[col], Q3[col] + self.multiplier * IQR[col])
            for col in df.columns
        }
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        mask = pd.Series(True, index=df.index)
        for col, (low, high) in self.bounds_.items():
            mask &= df[col].between(low, high)
        return df[mask]


def get_highly_correlated_pairs(df, threshold=0.9):
    """
    Returns pairs of features with correlation above the threshold.
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    return [
        (col1, col2, upper.loc[col1, col2])
        for col1 in upper.columns
        for col2 in upper.index
        if upper.loc[col1, col2] > threshold
    ]


def log_feature_variances(df):
    """
    Logs feature variances for inspection.
    """
    variances = df.var().sort_values(ascending=False)
    print("Feature variances:")
    print(variances)
    return variances


def remove_high_missing_features(df, threshold=0.5):
    """
    Removes features with missing values exceeding the threshold.
    """
    missing_ratio = df.isnull().mean()
    to_drop = missing_ratio[missing_ratio > threshold].index
    return df.drop(columns=to_drop)

