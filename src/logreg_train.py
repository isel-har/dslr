import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from utils.logistic_reg import FTLogisticRegression
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.linear_model import LogisticRegression


def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_csv>")
        return

    df = pd.read_csv(sys.argv[1])

    # Encode target variable
    le = LabelEncoder()
    transformed_y = le.fit_transform(df["Hogwarts House"])

    # Keep only numeric columns
    X = df.select_dtypes(include='number').drop(columns=["Index"], errors="ignore")

    # Drop highly correlated features
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    X = X.drop(columns=to_drop)

    # Fill missing values
    X = X.fillna(0)

    # Standardize features
    X = (X - X.mean()) / X.std()

    model = FTLogisticRegression(multi_class=True)
    # clf   = LogisticRegression(multi_class='ovr')
    try:
        model.train(X.to_numpy(), transformed_y)
        # clf.fit(X.to_numpy(), transformed_y)
    except TypeError:
        print("type error")
    except Exception as e:
        print("exception:", str(e))

    ym_pred = model.predict(X)


    # y_true = None
    # if "Hogwarts House" in df.columns:
    #     y_true = le.transform(df["Hogwarts House"])
    #  🔟 Convert numeric predictions back to original labels
    # house_predictions = le.inverse_transform(y_pred)
    
    # house_pred = le.inverse_transform(y_pred)

    # ✅ If true labels exist, evaluate accuracy
    # if y_true is not None:
    #     acc = accuracy_score(y_true, ym_pred)
    #     print(f"\nAccuracy: {acc * 100:.2f}%")
    #     print("\nClassification Report:")
    #     print(classification_report(y_true, ym_pred, target_names=le.classes_))
    #     print("\nConfusion Matrix:")
    #     print(confusion_matrix(y_true, ym_pred))
    # else:
    #     print("\n⚠️ No true labels found in test CSV — skipping accuracy evaluation.")
    # Print or save predictions
    

    # # Save all parameters to JSON
    # try:
    #     with open('logreg_params.json', 'w') as file:
    #         json.dump(params, file, indent=4)
    #     print("✅ Parameters saved successfully to 'logreg_params.json'")
    # except IOError as e:
    #     print("❌ Error writing file:", str(e))
    # except Exception as e:
    #     print("❌ Unexpected error:", str(e))

if __name__ == "__main__":
    main()


# import sys
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# from utils.logistic_reg import FTLogisticRegression
# import json
# # from sklearn.linear_model import LogisticRegression


# def main():
#     if len(sys.argv) != 2:
#         print("Usage: python script.py <path_to_csv>")
#         return

#     df = pd.read_csv(sys.argv[1])

#     # Encode target variable
#     le = LabelEncoder()
#     transformed_y = le.fit_transform(df["Hogwarts House"])

#     # Keep only numeric columns
#     X = df.select_dtypes(include='number').drop(columns=["Index"], errors="ignore")

#     # Drop highly correlated features
#     corr = X.corr().abs()
#     upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
#     to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
#     X = X.drop(columns=to_drop)

#     # Fill missing values
#     X = X.fillna(0)

#     # Standardize features
#     X = (X - X.mean()) / X.std()


#     model = FTLogisticRegression(multi_class=True)
#     params = {'to_drop':to_drop}
#     # clf   = LogisticRegression(multi_class='ovr')
#     try:
#         model.train(X.to_numpy(), transformed_y)
#         classes_ = le.classes_.tolist()
#         params['classes'] = classes_
#         for i, k in enumerate(classes_, start=0):
#             params[k] = {'weights':model.k_classes[i].weights.tolist(), 'bias':float(model.k_classes[i].bias)}
#     except TypeError as e:
#         print("type error:", str(e))
#     except Exception as e:
#         print("exception:", str(e))

#     try:
#         with open('logreg_params.json', 'w') as file:
#             json.dump(params, file, indent=4)
#         print("✅ Parameters saved successfully to 'logreg_params.json'")
#     except IOError as e:
#         print("❌ Error writing file:", str(e))
#     except Exception as e:
#         print("❌ Unexpected error:", str(e))

# if __name__ == "__main__":
#     main()
