import numpy as np

#TODO early stoping - L1/L2 - BGD/MBGD/SGD - learning animation?(maybe) - Adam/RMSprop/Momentum 

class LogisticRegression:

    def __init__(
        self,
        lr: float = 0.01,
        n_iters: int = 1000,
        batch_size: int | None = None,
        optimizer: str | None = None,
        penalty: str | None = None,
        random_seed: int = 42,
        early_stoping = False,
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
        self.W  = None
        self.b  = None
    
    def _sigmoid(self, z):
        #TODO maybe a clamp here to prevent overflow
        return 1 / (1 + np.exp(-z))

    def _loss(self, y, y_pred):
      

    def farward(self, X):
        return self._sigmoid(np.dot(X, self.W) + self.b)

    def fit(self, X, y):

        n_samples, n_features = X.Shape

        self.W = np.zeros(n_features) #TODO  a random initializer from a normal distribution 
        self.b = 0.0

    def predict(self, X):
        ...

    def score(self, X, y):
        #TODO return the mean accuracy score of self.predict() w.r.t. y
        ...

