import numpy as np
import os
from sklearn.model_selection import train_test_split


def _load(path):
    X = np.loadtxt(os.path.join(path, 'data.csv'), delimiter=',')
    y = np.loadtxt(os.path.join(path, 'target.csv'), delimiter=',')
    return X, y


def _save(path, prefix, X, y):
    np.savetxt(os.path.join(path, prefix + '_data.csv'), X, delimiter=',')
    np.savetxt(os.path.join(path, prefix + '_target.csv'), y, delimiter=',')


def _split(args):
    X, y = _load(args.data)
    X_train, X_val, y_train, y_val = train_test_split(X, y)
    _save(args.data, 'train', X_train, y_train)
    _save(args.data, 'val', X_val, y_val)


def init_subparser(parser):
    parser.add_argument('--data', type=str, required=True, help='Dataset directory')
    parser.set_defaults(func=_split)


if __name__ == '__main__':
    main()
