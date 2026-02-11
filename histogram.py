from utils.sfs import max_series, min_series,  mean_series#, variance_series
import matplotlib.pyplot as plt
import pandas as pd

from utils.sfs import mean_series

def main():
    if len(sys.argv) != 2:
        return

    try:
        df = pd.read_csv(sys.argv[1])
        df = df.drop(columns='Index')
        numeric_cols = df.select_dtypes(include='number').columns
        groups = df.groupby("Hogwarts House")[numeric_cols].agg(mean_series)

        diff = float("inf")
        best_course = None

        for course in groups.columns:
            d = max_series(groups[course]) - min_series(groups[course])
            if d < diff:
                diff = d
                best_course = course
        

        plt.figure(figsize=(8, 5))
        plt.bar(groups.index, groups[best_course])
        plt.xlabel("Hogwarts House")
        plt.ylabel("Average Score")
        plt.title(f"Most homogeneous course: {best_course}")
        plt.show()
    except Exception as e:
        print("exception:", str(e))
if __name__ == "__main__":
    main()
