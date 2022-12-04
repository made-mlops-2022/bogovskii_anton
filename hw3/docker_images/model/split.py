import argparse
import numpy as np
from sklearn.model_selection import train_test_split


def parse_args()
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path')
    return parser.parse_args()


def load_Xy(path):
    X = np.loadtxt(os.path.join(path, 'data.csv'), delimiter=', ')
    y = np.loadtxt(os.path.join(path, 'target.csv'), delimiter=', ')
    return X, y


def save_Xy(path, prefix, X, y):
    np.savetxt(os.path.join(path, prefix + '_data.csv'), X, delimiter=', ')
    np.savetxt(os.path.join(path, prefix + '_target.csv'), X, delimiter=', ')


def main():
    args = parse_args()
    X, y = load(args.dataset_path)

    X_train, X_val, y_train, y_val = train_test_split(X, y)
    save_Xy(args.dataset_path, 'train', X_train, y_train)
    save_Xy(args.dataset_path, 'val', X_val, y_val)


if __name__ == '__main__':
    main()
