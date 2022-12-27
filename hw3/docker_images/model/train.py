import argparse
import joblib
import numpy as np
import os
from sklearn.linear_model import LogisticRegression


def _train(args):
    X = np.loadtxt(os.path.join(args.data, 'train_data.csv'), delimiter=',')
    y = np.loadtxt(os.path.join(args.data, 'train_target.csv'), delimiter=',')

    os.makedirs(args.model, exist_ok=True)
    joblib.dump(
        LogisticRegression().fit(X, y),
        os.path.join(args.model, 'logreg.joblib'),
    )

def init_subparser(parser):
    parser.add_argument('--data', type=str, required=True, help='Dataset dir')
    parser.add_argument('--model', type=str, required=True, help='Model dir')
    parser.set_defaults(func=_train)
