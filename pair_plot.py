import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def preprocess(dataset: pd.DataFrame):

    selected_cols = dataset.select_dtypes(include="number").columns.to_list()
    selected_cols.remove("Index")

    selected_cols.remove("Defense Against the Dark Arts")
    selected_cols.remove("Care of Magical Creatures")

    selected_cols.append("Hogwarts House")

    dataset = dataset[selected_cols]
    dataset = dataset.dropna()
    return dataset


def main():

    if len(sys.argv) != 2:
        return
    try:
        df = pd.read_csv(sys.argv[1])
        proc_df = preprocess(df)
        g = sns.pairplot(
            proc_df,
            hue="Hogwarts House",
            diag_kind="hist",
            markers=".",
            corner=True,
            height=1,
            aspect=1.5,
        )
        plt.show()
    except Exception as e:
        print("exception:", str(e))


if __name__ == "__main__":
    main()
