import argparse
import json
import sys
import joblib
import numpy as np
import pandas as pd
import pathlib
from utils.impute import KNNImputer, SimpleImputer
from utils.logistic_regression import LogisticRegression
from utils.one_vs_rest_classifier import OneVsRestClassifier
from utils.scale import MinMaxScaler, StandardScaler
from utils.utils import DataFileAction

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        dest="model_file",
        help="the model file.",
        action=DataFileAction,
    )

    parser.add_argument(
        "test_dataset_file",
        type=pathlib.Path,
        help="the dataset file.",
        action=DataFileAction,
    )

    args = parser.parse_args()

    model = joblib.load(args.model_file)

    imputer = model["imputer"]
    scaler = model["scaler"]
    ovr = model["ovr"]

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

    X = imputer.transform(X)
    X = scaler.transform(X)
    y_pred = ovr.predict(X)

    with open("houses.csv", "w+") as pf:
        pf.write("Index,Hogwarts House\n")
        for i in range(0, len(y_pred)):
            pf.write(f"{i},{y_pred[i]}\n")
