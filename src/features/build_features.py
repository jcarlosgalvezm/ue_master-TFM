from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline


class Sparse2Array(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.toarray()

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


def get_preprocessing_steps():

    numeric_cols = ['horas_formacion', 'indice_desarrollo_ciudad']
    categorical_cols = [
        'ultimo_nuevo_trabajo',
        'tamano_compania',
        'experiencia',
        'educacion',
        'universidad_matriculado',
        'nivel_educacion',
        'experiencia_relevante'
        ]

    numeric_tf = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_tf = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(drop='first'))
    ])

    transformer = ColumnTransformer(
        transformers=[
            ('num', numeric_tf, numeric_cols),
            ('cat', categorical_tf, categorical_cols),
        ])

    return make_pipeline(transformer, Sparse2Array(), PCA())
