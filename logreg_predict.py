import argparse
import json
import sys

import numpy as np

from utils.logistic_regression import LogisticRegression
from utils.one_vs_rest_classifier import OneVsRestClassifier
from utils.scale import MinMaxScaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=sys.argv[0], usage="python %(prog)s [weights file] [dataset file]"
    )
    parser.add_argument(
        "weights_file",
        nargs=1,
        type=argparse.FileType("r"),
        help="the models weights file.",
    )
    parser.add_argument(
        "dataset_file", nargs=1, type=argparse.FileType("r"), help="the dataset file."
    )
    args = parser.parse_args()

    with open(args.weights_file[0].name) as wf:
        artifact = json.load(wf)

    with open(args.dataset_file[0].name) as df:
        test_dataset = None  # use pandas to load dataset?

    X = test_dataset.tonumpy()  # TODO slice it?
    scaler = MinMaxScaler()
    scaler.data_min_ = np.array(
        artifact["scaler"]["min"]
    )  # TODO add set_params to all scalers
    scaler.data_max_ = np.array(artifact["scaler"]["max"])

    estimators = []
    for cls in artifact["classes"]:
        lr = LogisticRegression()
        w = artifact["weights"][str(cls)]
        lr.W = np.array(w["W"])  # TODO add sample_weights initializer to the LR model
        estimators.append(lr)

    ovr = OneVsRestClassifier(None)
    ovr.estimators_ = estimators
    ovr.classes_ = np.array(artifact["classes"])
    y_pred = ovr.predict(X)

    with open("houses.csv", "w+") as pf:
        pf.write("Index,Hogwarts House\n")
        for i in range(0, len(y_pred)):
            f.write(f"{i},{y_pred[i]}\n")
