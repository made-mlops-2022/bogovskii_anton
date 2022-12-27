import numpy as np
import os
import shutil


def _one_hot_encode_last_feature(X):
    X_numeric = X[:,:-1]
    X_categorical = X[:,-1]
    X_categorical_encoded = np.zeros((X.shape[0], int(X_categorical.max() + 1)))
    return np.concatenate((X_numeric, X_categorical_encoded), axis=1)


def _preprocess(args):
    X = np.loadtxt(os.path.join(args.raw, 'data.csv'), delimiter=',')
    X_processed = _one_hot_encode_last_feature(X)

    os.makedirs(args.processed, exist_ok=True)
    np.savetxt(os.path.join(args.processed, 'data.csv'), X_processed, delimiter=',')
    shutil.copyfile(os.path.join(args.raw, 'target.csv'), os.path.join(args.processed, 'target.csv'))


def init_subparser(parser):
    parser.add_argument('--raw', type=str, required=True, help='Raw dataset dir')
    parser.add_argument('--processed', type=str, required=True, help='Processed dataset dir')
    parser.set_defaults(func=_preprocess)
