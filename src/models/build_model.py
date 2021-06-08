from sklearn.linear_model import LogisticRegression

DEFAULT_PARAMS = {
    'random_state': 288
}


def build_model(**kwargs):
    hyperparameters = kwargs if kwargs else DEFAULT_PARAMS
    model = LogisticRegression(**hyperparameters)
    return model
