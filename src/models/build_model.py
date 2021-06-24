from sklearn.svm import SVC

DEFAULT_PARAMS = dict(C=100, gamma='auto', kernel='rbf', random_state=288)


def build_model(**kwargs):
    hyperparameters = kwargs if kwargs else DEFAULT_PARAMS
    model = SVC(**hyperparameters)
    return model
