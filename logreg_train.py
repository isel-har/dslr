from utils.logistic_regression import LogisticRegression
from utils.one_vs_rest_classifier import OneVsRestClassifier
from utils.scale import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from utils.utils import train_test_split

if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    scaler = MinMaxScaler().fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression( penalty="l", batch_size=None, optimizer=None, verbose=True, early_stopping=False)
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
