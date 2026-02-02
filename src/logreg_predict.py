# from scipy.special import expit
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils.logistic_reg import FTLogisticRegression
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import json
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_csv>")
        return

    df = pd.read_csv(sys.argv[1])
    X = df.select_dtypes(include='number').drop(columns=["Index"], errors="ignore")

    X = X.apply(lambda col: col.fillna(col.mean()), axis=0)
    X = X.fillna(0)

    try:
        with open('./logreg_params.json', 'r') as content:
            params = json.loads(content.read())
    except IOError as e:
        print(f"Error reading string to file: {e}")

    to_drop  = params['to_drop']
    X = X.drop(columns=to_drop)
    X = X.drop(columns='Hogwarts House')
    X = (X - X.mean()) / X.std()


    classes_ = params['classes']
    model  = FTLogisticRegression(multi_class=True)
    model.class_num = len(classes_)

    PClass = FTLogisticRegression().Pclass

    for class_ in classes_:
        weights = np.array(params[class_]['weights'])
        bias    = np.float64(params[class_]['bias'])

        print(weights.shape)    
        kclass = PClass(n_features=0,weights=weights, bias=bias)
        model.k_classes.append(kclass)

    # le = LabelEncoder()
    preds_class = model.predict(X)

    house_pred = np.array([classes_[p]  for p in preds_class])

    output = pd.DataFrame({
        "Index": df.get("Index", range(len(house_pred))),
        "Hogwarts House": house_pred
    })
    output.to_csv("houses.csv", index=False)
    print("\nPredictions saved to houses.csv ✅")

if __name__ == "__main__":
    main()