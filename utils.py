import pandas as pd


def feature_discretizaiton(X, key, bins):
    labels = range(len(bins)-1)
    print(X[key])
    X[key] = pd.cut(X[key], bins=bins, labels=labels)
    X[key] = X[key].cat.codes
    return X
