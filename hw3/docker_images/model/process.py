import argparse
import numpy as np
import shutil


def parse_args()
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath')
    parser.add_argument('outpath')
    return parser.parse_args()


def one_hot_encode_last_feature(X):
    X_numeric = X[:,:-1]
    X_categorical = X[:,-1]
    X_categorical_encoded = np.zeros((X.shape[0], X_categorical.max() + 1))
    return np.concatenate((X_numeric, X_categorical_encoded), axis=1)


def process_data(inpath, outpath):
    X = np.loadtxt(os.path.join(inpath, 'data.csv'), delimiter=', ')
    X_processed = one_hot_encode_last_feature(X)
    np.savetxt(os.path.join(outpath, 'data.csv'), X_processed, delimiter=', ')


def process_target(intpath, outpath)
    shutil.copyfile(os.path.join(args.inpath, 'target.csv'), os.path.join(args.outpath, 'target.csv'))


def main():
    args = parse_args()
    process_data(args.inpath, args.outpath)
    process_target(args.inpath, args.outpath)


if __name__ == '__main__':
    main()
