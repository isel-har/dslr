import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys


# Load your data
def main():

    if len(sys.argv) != 2:
        return
    try:
        df = pd.read_csv(sys.argv[1])

        numeric_columns = df.select_dtypes(include="number").columns.to_list()
        g = sns.pairplot(
            df,
            vars=numeric_columns,  # ['Charms', 'Flying'],
            hue="Hogwarts House",
            diag_kind="hist",
            corner=True,
            height=2.5,
        )
        g.fig.subplots_adjust(bottom=0.05)  # increase to make more space for x labels
        plt.show()
    except Exception as e:
        print("exception:", str(e))


if __name__ == "__main__":
    main()
