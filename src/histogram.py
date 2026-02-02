from utils.sfs import mean_series
import matplotlib.pyplot as plt
import pandas as pd
import sys

def main():

    if len(sys.argv) != 2: return
    df = pd.read_csv(sys.argv[1])

    selected_columns = df.select_dtypes(include='number').columns.to_list()
    groups = df.groupby("Hogwarts House")[selected_columns].agg(mean_series)

    diff = float('inf')
    smallest_score = ""
    for name, values in groups.items():
        v = max(values) - min(values)
        if diff > v:
            diff = v
            smallest_score = name

    # values = groups[smallest_score].values

    plt.figure(figsize=(8, 5))
    plt.bar(groups.index, groups.loc[:, smallest_score])
    plt.xlabel("Hogwarts House")
    plt.ylabel("Average Score")
    plt.title(f"Homogeneous Score Distribution for {smallest_score}")
    plt.show()

if __name__ == "__main__":
    main()