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


