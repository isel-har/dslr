from scipy.special import expit
import numpy as np


class FTLogisticRegression:
    class Pclass:
        def __init__(self, n_features, weights=None, bias=float(0.0)):
            self.weights = np.zeros(n_features) if weights is None else weights
            self.bias = bias

    def __init__(
        self, learning_rate=0.05, epochs=1000, optimizer="batch", multi_class=False
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.k_classes = list()
        self.optimizer_ref = (
            self.stochastic_gd if optimizer == "stochastic" else self.batch_gd
        )
        self.sample_size = 0
        self.multi_class = multi_class
        self.class_num = 1

    def batch_gd(self, x, y_binary, c_object: Pclass):
        for _ in range(self.epochs):
            # Compute predictions (vectorized)
            z = np.dot(x, c_object.weights) + c_object.bias
            y_pred = expit(z)

            # Compute gradients
            dw = np.dot(x.T, (y_pred - y_binary)) / self.sample_size
            db = np.mean(y_pred - y_binary)

            # Update parameters
            c_object.weights -= self.learning_rate * dw
            c_object.bias -= self.learning_rate * db

    def stochastic_gd(self, x, y_binary):
        ###
        for _ in range(self.epochs):
            # Compute predictions (vectorized)
            z = np.dot(x, self.weights) + self.bias
            y_pred = expit(z)

            # Compute gradients
            dw = np.dot(x.T, (y_pred - y_binary))
            db = np.mean(y_pred - y_binary)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def train(self, x, y):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("x and y must be numpy arrays.")

        self.sample_size, n_features = x.shape  ## extraction of sizes

        self.class_num = y.max() + 1 if self.multi_class else 1
        for _ in range(self.class_num):
            print(f"Init class {_} paramters.")
            self.k_classes.append(self.Pclass(n_features))

        for k in range(self.class_num):
            print(f"Training class {k}...")
            y_binary = np.where(y == k, 1, 0)
            self.optimizer_ref(x, y_binary, self.k_classes[k])

    def predict_proba(self, x):
        """Return predicted probabilities."""
        if self.multi_class:
            zs = []
            for k in range(self.class_num):
                weights = self.k_classes[k].weights
                bias = self.k_classes[k].bias
                z = np.dot(x, weights) + bias
                zs.append(expit(z))
            # shape (n_classes, n_samples)
            return np.array(zs).T  # Transpose to (n_samples, n_classes)
        else:
            weights = self.k_classes[0].weights
            bias = self.k_classes[0].bias
            z = np.dot(x, weights) + bias
            return expit(z)  # shape (n_samples,)

    def predict(self, x):
        """Return class predictions."""
        y_pred = self.predict_proba(x)
        if self.multi_class:
            # Choose the class with highest probability
            return np.argmax(y_pred, axis=1)
        else:
            return (y_pred >= 0.5).astype(int)
