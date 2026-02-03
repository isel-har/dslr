import numpy as np

#TODO early stoping - L1/L2 - BGD/MBGD/SGD - learning animation?(maybe) - Adam/RMSprop/Momentum 

class LogisticRegression:

    def __init__(self, lr=0.1, n_iters = 100, method='BGD', verbose=0, random_state=42, epsilon=1e-4, n_jobs=None):
        self.lr = lr
        self.W  = None
        self.b  = None
        self.n_iters = n_iters
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
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

#TODO Running the classifiers on parallel - some error checks - 
class OneVsRestClassifier:

    def __init__(self, estimator, *, n_jobs=None, verbose=0):
        self.classes_ =  None
        self.models  = []
        ...

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.models_ = []

        for c in self.classes_:
            y_bin = (y == c).astype(int)

            model = LogisticRegression()
            model.fit(X, y_bin)
            self.models.append(model)
        
        return self
        
    def predict_probabilities(self): #TODO iterate over all the models and run predict()

    def predict(self, X): #TODO return the name of the class with highest probability self.classes_[np.argmaxy(_pred, axis=1)] 
        ...


