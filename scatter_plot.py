import sys

import matplotlib.pyplot as plt
import pandas as pd

from utils.sfs import correlation_


def main():
    if len(sys.argv) != 2:
        return

    try:
        df = pd.read_csv(sys.argv[1])
        numeric_df = df.select_dtypes(include="number")
        columns = numeric_df.columns.to_list()

        max_cor = 0
        score_x = ""
        score_y = ""

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                x_vals = numeric_df[columns[i]].to_list()
                y_vals = numeric_df[columns[j]].to_list()
                cor = correlation_(x_vals, y_vals)

                if abs(cor) > abs(
                    max_cor
                ):  # consider absolute value for strongest correlation
                    max_cor = cor
                    score_x = columns[i]
                    score_y = columns[j]

        plt.scatter(numeric_df[score_x], numeric_df[score_y], color="orange")
        plt.xlabel(score_x)
        plt.ylabel(score_y)
        plt.title("Scatter Plot")
        plt.show()
    except Exception as e:
        print("exception:", str(e))


if __name__ == "__main__":
    main()
