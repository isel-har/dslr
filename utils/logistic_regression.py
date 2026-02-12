import pickle

import numpy as np


class LogisticRegression:
    """Binary Logistic Regression Model

    params:
    ------
        lr: float, Learning rate (defaut=0.01).
        n_iters: int = 1000, number of epochs.
        batch_size: int | None = None, size of each batch in mini-batch and stochastic gradient descent if None the whole batch is used.
        optimizer: str | None = None, gradient descent optimizer, options: "adam" | "rmsprop" | "momentum" | None.
        penalty: str | None = None, regularization technique, options: "l1" | "l2" | None.
        early_stoping=False, stop training when the loss stops decreasing by `tol` for `patience` epochs.
        random_seed: int = 42, random seed to initialize the weights matrix.
        patience: int = 10, defines how many epochs to keep training when the loss stops decreasing.
        C: float = 1., the inverse of the regularization strenght.
        tol: float = 1e-7,
        verbose: bool = False, if `True` training progress printed to stdout.
        epsilon: float = 1e-9, very small value used to prevent dividing by 0 .

    attrs:
    -----

    returns:
    -----
        self, Fitted estimator.
    """

    def __init__(
        self,
        lr=0.01,
        n_iters=1000,
        batch_size=None,
        optimizer=None,
        penalty=None,
        C=1.0,
        early_stopping=False,
        tol=1e-7,
        patience=10,
        random_seed=42,
        verbose=False,
        epsilon=1e-9,
    ):

        assert penalty in ("l1", "l2", None), f"unknown penalty: {penalty}"
        assert optimizer in ("adam", "momentum", "rmsprop", None), (
            f"unknown optimizer: {optimizer}"
        )

        self.lr = lr
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.penalty = penalty
        self.C = C

        self.early_stopping = early_stopping
        self.patience = patience
        self.tol = tol
        self.verbose = verbose
        self.epsilon = epsilon

        self.random_seed = random_seed
        self.W = None
        self.is_fitted = False

        self.v = None
        self.s = None

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _add_bias(self, X):
        return np.c_[np.ones(X.shape[0]), X]

    def _loss(self, y, y_pred):
        m = len(y)
        loss = -(1 / m) * np.sum(
            y * np.log(y_pred + self.epsilon)
            + (1 - y) * np.log(1 - y_pred + self.epsilon)
        )

        if self.penalty == "l1":
            loss += (1 / self.C) * np.sum(np.abs(self.W[1:]))

        elif self.penalty == "l2":
            loss += (1 / (2 * self.C)) * np.sum(self.W[1:] ** 2)

        return loss

    def _gradient(self, X, y, y_pred):
        m = len(y)
        dw = (1 / m) * X.T @ (y_pred - y)

        if self.penalty == "l1":
            dw[1:] += (1 / self.C) * np.sign(self.W[1:])

        elif self.penalty == "l2":
            dw[1:] += (1 / self.C) * self.W[1:]

        return dw

    def _update_weights(self, dw, t):
        if self.optimizer is None:
            self.W -= self.lr * dw

        elif self.optimizer == "momentum":
            beta = 0.9
            self.v = beta * self.v + (1 - beta) * dw
            self.W -= self.lr * self.v

        elif self.optimizer == "rmsprop":
            beta = 0.9
            self.s = beta * self.s + (1 - beta) * (dw**2)
            self.W -= self.lr * dw / (np.sqrt(self.s) + self.epsilon)

        elif self.optimizer == "adam":
            beta1, beta2 = 0.9, 0.999

            self.v = beta1 * self.v + (1 - beta1) * dw
            self.s = beta2 * self.s + (1 - beta2) * (dw**2)

            v_hat = self.v / (1 - beta1**t)
            s_hat = self.s / (1 - beta2**t)

            self.W -= self.lr * v_hat / (np.sqrt(s_hat) + self.epsilon)

    def fit(self, X, y):
        np.random.seed(self.random_seed)

        X = self._add_bias(X)
        n_samples, n_features = X.shape

        self.W = np.zeros(n_features)
        self.v = np.zeros_like(self.W)
        self.s = np.zeros_like(self.W)

        best_loss = np.inf
        wait = 0
        self.losses_ = []

        for epoch in range(1, self.n_iters + 1):
            if self.batch_size is None:
                X_batch, y_batch = X, y
            else:
                idx = np.random.choice(n_samples, self.batch_size, replace=False)
                X_batch, y_batch = X[idx], y[idx]

            y_pred = self._sigmoid(X_batch @ self.W)

            loss = self._loss(y_batch, y_pred)

            dw = self._gradient(X_batch, y_batch, y_pred)

            self._update_weights(dw, epoch)

            self.losses_.append(loss)
            if self.early_stopping:
                if best_loss - loss > self.tol:
                    best_loss = loss
                    wait = 0
                else:
                    wait += 1

                if wait >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss={loss:.6f}")

        self.is_fitted = True
        return self

    def predict_proba(self, X):
        X = self._add_bias(X)
        return self._sigmoid(X @ self.W)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

    def save_model(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load_model(self, filename):
        with open(filename, "rb") as f:
            self.__dict__ = pickle.load(f)
