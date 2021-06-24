import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline


class OutliersTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method='drop', factor=1.5, exclude_cols=None):
        self.method = method
        self.factor = factor
        self.exclude_cols = exclude_cols

    def _outlier_removal(self, X, y=None):
        X = pd.Series(X).copy()
        if X.name not in self.exclude_cols:
            q1 = X.quantile(0.25)
            q3 = X.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (self.factor * iqr)
            upper_bound = q3 + (self.factor * iqr)
            X.loc[((X < lower_bound) | (X > upper_bound))] = np.nan
        return pd.Series(X)

    def _outlier_cap(self, X, y=None):
        X = pd.Series(X).copy()
        if X.name not in self.exclude_cols:
            q1 = X.quantile(0.25)
            q3 = X.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (self.factor * iqr)
            upper_bound = q3 + (self.factor * iqr)
            X.loc[X < lower_bound] = lower_bound
            X.loc[X > upper_bound] = upper_bound
        return pd.Series(X)

    def fit(self, X, y=None):
        self.exclude_cols = \
            set(self.exclude_cols) if self.exclude_cols else set()
        return self

    def transform(self, X, y=None):
        if self.method == 'drop':
            return X.apply(self._outlier_removal)
        elif self.method == 'cap':
            return X.apply(self._outlier_cap)
        else:
            return X

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


def get_preprocessing_steps():

    transformer = ColumnTransformer(transformers=[
        ('num', Pipeline(steps=[
                ('outliers', OutliersTransformer(method='cap',
                    exclude_cols=['indice_desarrollo_ciudad'])),
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', MinMaxScaler())
        ]), ['indice_desarrollo_ciudad', 'horas_formacion']),
        ('cat', Pipeline(steps=[
                ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))
        ]), ['ciudad', 'genero', 'nivel_educacion', 'experiencia',
            'tamano_compania', 'ultimo_nuevo_trabajo'])
    ])

    return make_pipeline(transformer, PCA(n_components=110))
