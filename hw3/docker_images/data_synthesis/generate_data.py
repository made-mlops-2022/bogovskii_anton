import argparse
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser('Generate data for dataset')
    parser.add_argument('dataset', help='dataset directory')
    parser.add_argument('--n-samples', type=int, default=1500)
    return parser.parse_args()


def generate_data(n_samples):
    coeffs = np.linalg.norm(np.fromiter([3, 100, -1, 3, -2], dtype=float), 1)

    X_numeric = np.random.normal(size=(n_samples, 2))
    X_categorical = np.random.randint(0, 3, size=(n_samples, 1))
    X = np.concatenate((X_numeric, X_categorical), axis=1)

    X_categorical_encoded = np.zeros((n_samples, 3))
    X_categorical_encoded[:,X_categorical[:,0]] = 1
    X_encoded = np.concatenate((X_numeric, X_categorical_encoded), axis=1)

    y = (coeffs.reshape(1, -1) * X_encoded + np.random.normal(0, .1, X_encoded.shape) > 0).sum(axis=1)

    return X, y


def save_data(X, y, path):
    os.makedirs(path, exist_ok=True)
    np.savetxt(os.path.join(path, 'data.csv'), X, delimiter=',')
    np.savetxt(os.path.join(path, 'target.csv'), y, delimiter=',')


def main():
    args = parse_args()
    X, y = generate_data(args.n_samples)
    save_data(X, y, args.dataset)


if __name__ == '__main__':
    main()
