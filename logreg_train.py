import argparse
import json
import os
import pathlib
import sys
import joblib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from utils.impute import KNNImputer, SimpleImputer
from utils.logistic_regression import LogisticRegression
from utils.one_vs_rest_classifier import OneVsRestClassifier
from utils.scale import MinMaxScaler, RobustScaler, StandardScaler
from utils.utils import (
    DataFileAction,
    PositiveFloatAction,
    PositiveIntAction,
    train_test_split,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.suggest_on_error = True
    parser.add_argument(
        "train_dataset_file",
        type=pathlib.Path,
        help="the dataset file.",
        action=DataFileAction,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-p",
        "--plot",
        help="display the loss plot for each model after training",
        action="store_true",
    )
    parser.add_argument(
        "-e",
        "--early-stop",
        help="stop training when the loss stops decreasing",
        action="store_true",
    )
    parser.add_argument(
        "--lr",
        help="(default=%(default)s) learning rate: model step size in gradient descent (be careful, a large learning rate will prevent convergenece !)",
        type=float,
        default=0.01,
        action=PositiveFloatAction,
    )

    parser.add_argument(
        "--penalty",
        help="(default=%(default)s) regularization method for model params",
        choices=["l1", "l2"],
        default=None,
    )
    parser.add_argument(
        "--optimizer",
        help="(default=%(default)s) gradient descent optimizer",
        choices=["adam", "rmsprop", "momentum"],
        default=None,
    )
    parser.add_argument(
        "--niters",
        help="(default=%(default)s) number of training iterations",
        type=int,
        default=int(1000),
        action=PositiveIntAction,
    )
    parser.add_argument(
        "--batch-size",
        help="(default=%(default)s) batch size for mini-batch and stochastic gradient descent, if None the whole training set is used",
        choices=[16, 32, 64, 128],
        type=int,
        default=None,
    )

    args = parser.parse_args()

    useless_cols = [
        "Index",
        "First Name",
        "Last Name",
        "Birthday",
        "Best Hand",
        "Arithmancy",
        "Defense Against the Dark Arts",
        "Care of Magical Creatures",
    ]

    test_dataset = (
        pd.read_csv(args.train_dataset_file).drop(columns=useless_cols).to_numpy()
    )
    X, y = test_dataset[:, 1:], test_dataset[:, 0:1].flatten()

    imputer = SimpleImputer(strategy="mean").fit(X)
    X = imputer.transform(X)

    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    model = LogisticRegression(
        lr=args.lr,
        penalty=args.penalty,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        verbose=args.verbose,
        early_stopping=args.early_stop,
        n_iters=args.niters,
    )
    ovr = OneVsRestClassifier(model, n_jobs=4, verbose=args.verbose)
    ovr.fit(X, y)

    artifact = {
        "imputer": {"statistics": imputer.statistics_.tolist()},
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "classes": ovr.classes_.tolist(),
        "weights": {},
    }

    model = {"imputer": imputer, "scaler": scaler, "ovr": ovr}

    joblib.dump(model, "model.pkl")

    if args.plot:
        import matplotlib.pyplot as plt

        losses = ovr.get_models_losses()
        fig, axs = plt.subplots(nrows=4, ncols=1, layout="constrained")
        for k in losses.keys():
            axs[k].plot(list(range(len(losses[k]))), losses[k])
            axs[k].set_xlabel("Epochs")
            axs[k].set_ylabel("Loss")
            axs[k].set_title(f"Model {k}")
            axs[k].grid(True)
        plt.show()
