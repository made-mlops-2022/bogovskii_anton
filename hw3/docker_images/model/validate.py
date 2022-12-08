import joblib
import numpy as np
import os
from sklearn.linear_model import LogisticRegression


def _validate(args):
    model = joblib.load(os.path.join(args.model, 'logreg.joblib'))
    X = np.loadtxt(os.path.join(args.data, 'val_data.csv'), delimiter=',')
    y = np.loadtxt(os.path.join(args.data, 'val_target.csv'), delimiter=',')

    with open(os.path.join(args.model, 'validation_accuracy.txt'), 'w') as f:
        f.write(str((y == model.predict(X)) / len(y)) + '\n')


def init_subparser(parser):
    parser.add_argument('--data', type=str, required=True, help='Dataset dir')
    parser.add_argument('--model', type=str, required=True, help='Model dir')
    parser.set_defaults(func=_validate)
