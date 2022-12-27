import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORY_INDS = [1, 2, 5, 6, 8, 10, 11, 12]


class BaselineModel(BaseEstimator):
    def __init__(self, num_features, category_inds):
        self.num_features_ = num_features
        self.category_inds_ = category_inds
        self.num_inds_ = list(set(range(num_features)) - set(category_inds))

        self.encoder_ = OneHotEncoder(drop='first', sparse=False)
        self.scaler_ = StandardScaler()
        self.regressor_ = LogisticRegression()

    def fit(self, X, y):
        X_categorical = self.encoder_.fit_transform(X[:, self.category_inds_])
        X_numeric = self.scaler_.fit_transform(X[:, self.num_inds_])
        X_transformed = np.concatenate((X_categorical, X_numeric), axis=1)
        self.regressor_.fit(X_transformed, y)
        return self

    def predict(self, X):
        X_categorical = self.encoder_.transform(X[:,self.category_inds_])
        X_numeric = self.scaler_.transform(X[:,self.num_inds_])
        X_transformed = np.concatenate([X_categorical, X_numeric], axis=1)
        return self.regressor_.predict(X_transformed)

    def save(self, path):
        state ={
            'num_features':  self.num_features_,
            'category_inds': self.category_inds_,
            'encoder':       self.encoder_,
            'scaler':        self.scaler_,
            'regressor':     self.regressor_,
        }
        joblib.dump(state, path)

    @staticmethod
    def load(path):
        state = joblib.load(path)
        model = BaselineModel(state['num_features'], state['category_inds'])
        model.encoder_ = state['encoder']
        model.scaler_ = state['scaler']
        model.regressor_ = state['regressor']
        return model


def load_dataset(path):
    Xy = pd.read_csv(path).values
    return Xy[:,:-1], Xy[:,-1]


def train_model(model_path, X, y, category_inds):
    assert max(category_inds) < X.shape[1]
    BaselineModel(X.shape[1], category_inds).fit(X, y).save(model_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path')
    parser.add_argument('model_path')
    args = parser.parse_args()

    X, y = load_dataset(args.dataset_path)
    train_model(args.model_path, X, y, CATEGORY_INDS)


if __name__ == '__main__':
    main()
