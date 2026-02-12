import argparse
import json
import sys

import numpy as np
import pandas as pd

from utils.impute import KNNImputer, SimpleImputer
from utils.logistic_regression import LogisticRegression
from utils.one_vs_rest_classifier import OneVsRestClassifier
from utils.scale import MinMaxScaler, StandardScaler
from utils.utils import DataFileAction

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        dest="weights_file",
        help="the models weights file.",
        required=True,
        action=DataFileAction,
    )
    parser.add_argument(
        "--data",
        dest="test_dataset_file",
        help="the dataset file.",
        required=True,
        action=DataFileAction,
    )
    args = parser.parse_args()
    print(args)
    with open(args.weights_file) as wf:
        artifact = json.load(wf)

    useless_cols = [
        "Index",
        "Hogwarts House",
        "First Name",
        "Last Name",
        "Birthday",
        "Best Hand",
        "Arithmancy",
        "Defense Against the Dark Arts",
        "Care of Magical Creatures",
    ]
    X = pd.read_csv(args.test_dataset_file).drop(columns=useless_cols).to_numpy()

    imputer = SimpleImputer(strategy="mean")
    imputer.statistics_ = np.array(artifact["imputer"]["statistics"])
    X = imputer.transform(X)

    scaler = StandardScaler()
    scaler.mean_ = np.array(artifact["scaler"]["mean"])
    scaler.scale_ = np.array(artifact["scaler"]["scale"])

    estimators = []
    for cls in artifact["classes"]:
        lr = LogisticRegression()
        w = artifact["weights"][str(cls)]
        lr.W = np.array(w["W"])
        estimators.append(lr)

    ovr = OneVsRestClassifier(None)
    ovr.estimators_ = estimators
    ovr.classes_ = np.array(artifact["classes"])
    y_pred = ovr.predict(X)

    with open("houses.csv", "w+") as pf:
        pf.write("Index,Hogwarts House\n")
        for i in range(0, len(y_pred)):
            pf.write(f"{i},{y_pred[i]}\n")
