import numpy as np

# TODO early stoping - L1/L2 - BGD/MBGD/SGD - learning animation?(maybe) - Adam/RMSprop/Momentum


class LogisticRegression:
    """Binary Logistic Regression Model

    params:
    ------
        lr: float, Learning rate (defaut=0.01).
        n_iters: int = 1000, number of epochs.
        batch_size: int | None = None, size of each batch in mini-batch and stochastic gradient descent if None the whole bath is used.
        optimizer: str | None = None, gradient descent optimizer, options: "adam" | "rmsprop" | "momentum" | None.
        penalty: str | None = None, regularization technique, options: "l1" | "l2" | None.
        early_stoping=False, stop training when the loss stops decreasing by `tol` for `patience` epochs.
        random_seed: int = 42, random seed to initialize the weights matrix.
        patience: int = 10, defines how many epochs to keep training when the loss stops decreasing.
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
        lr: float = 0.01,
        n_iters: int = 1000,
        batch_size: int | None = None,
        optimizer: str | None = None,
        penalty: str | None = None,
        random_seed: int = 42,
        early_stoping=False,
        patience: int = 10,
        verbose: bool = False,
        epsilon: float = 1e-9,
    ):

        self.lr = lr
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.optimizer = optimizer

        self.penalty = penalty
        self.early_stopping = early_stopping
        self.tol = tol
        self.patience = patience
        self.verbose = verbose
        self.W = None
        self.b = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _loss(self, y, y_pred):

        m = len(y)
        loss = -(1 / m) * np.sum(
            y * np.log(y_pred + self.epsilon)
            + (1 - y) * np.log((1 - y_pred) + self.epsilon)
        )

        if self.penalty == "l1":
            ...

        elif self.penalty == "l2":
            ...

        return loss

    def _gradient(self, y, y_pred):
        m = len(y)

        dw = (1 / m) * (X.T @ (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)

        if self.penalty == "l1":
            ...

        elif self.penalty == "l2":
            ...

        return dw, wb

    def _farward(self, X):
        return self._sigmoid(np.dot(X, self.W) + self.b)

    def _update_weights(self, dw, db):

        if self.optimizer == None:
            self.W -= self.lr * dw
            self.b -= self.lr * db

        elif self.optimizer == "adam":
            ...

        elif self.optimizer == "rmsprop":
            ...

        elif self.optimizer == "momentum":
            ...

    def fit(self, X, y):

        wait = 0
        n_samples, n_features = X.Shape

        self.W = np.random.rand(X.shape[1])
        self.b = 0.0
        prev_loss = np.inf

        for epoch in range(int(self.n_inters)):
            y_pred = self._forward(X)
            loss = self._loss(y, y_pred)

            if self.early_stoping:
                if prev_loss - loss > self.tol:
                    prev_loss = loss
                    wait = 0
                else:
                    wait += 1

                if wait >= self.patience:
                    break
        return self

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def predict_proba(self, X):
        return self._sigmoid(np.dot(X, self.W) + self.b)

    def score(self, X, y):
        # TODO return the mean accuracy score of self.predict() w.r.t. y
        ...

    def save_model(self, filename): ...

    def load_model(self, filename): ...
