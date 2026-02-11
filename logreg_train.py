import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

from utils.impute import KNNImputer, SimpleImputer
from utils.logistic_regression import LogisticRegression
from utils.one_vs_rest_classifier import OneVsRestClassifier
from utils.scale import MinMaxScaler, RobustScaler, StandardScaler
from utils.utils import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=sys.argv[0], usage="python %(prog)s [train dataset file]"
    )
    parser.add_argument(
        "test_dataset_file",
        nargs=1,
        type=argparse.FileType(
            "r"
        ),  # this shit keeps the file open btw, so avoid it or close after reading in a try/catch
        help="the dataset file.",
    )
    args = parser.parse_args()

    test_dataset = pd.read_csv(args.test_dataset_file[0].name).to_numpy()
    X, y = test_dataset[:, 6:], test_dataset[:, 1:2].flatten()
    imputer = KNNImputer().fit(X)
    X = imputer.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    model = LogisticRegression(
        penalty="l2", batch_size=64, optimizer=None, verbose=True, early_stopping=True
    )
    ovr = OneVsRestClassifier(model, n_jobs=4)
    ovr.fit(X_train, y_train)

    print(
        "train Accuracy: {} %".format(
            accuracy_score(y_train, ovr.predict(X_train)) * 100
        )
    )
    print(
        "test Accuracy:  {} %".format(accuracy_score(y_test, ovr.predict(X_test)) * 100)
    )

    # artifact = {
    #     "scaler": {
    #         "min": scaler.data_min_.tolist(),
    #         "max": scaler.data_max_.tolist(),
    #     },
    #     "classes": ovr.classes_.tolist(),
    #     "weights": {},
    # }

    # for cls, estimator in zip(ovr.classes_, ovr.estimators_):
    #     artifact["weights"][str(cls)] = {
    #         "W": estimator.W.tolist(),
    #     }

    # with open("models/weights.json", "w") as f:
    #     json.dump(artifact, f, indent=2)
