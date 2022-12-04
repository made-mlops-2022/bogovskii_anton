import argparse
import joblib
import numpy as np
import os
from sklearn.linear_model import LogisticRegression


def load_Xy_train(path):
    X = np.loadtxt(os.path.join(path, 'train_data.csv'), delimiter=', ')
    y = np.loadtxt(os.path.join(path, 'train_target.csv'), delimiter=', ')
    return X, y


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path')
    parser.add_argument('model_path')
    return parser.parse_args()


def main():
    args = parse_args()

    joblib.dump(
        LogisticRegression().fit(*load_Xy_train(args.dataset_path)),
        os.path.join(args.model_path, 'logreg.joblib'),
    )


if __name__ == '__main__':
    main()
