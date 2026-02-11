# from utils.csvt import CSVAnalyzer
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys

import pandas as pd

import utils.sfs as usfs


def main():

    if len(sys.argv) != 2:
        return

    try:
        df = pd.read_csv(sys.argv[1])
        selected_types = df.select_dtypes(include="number")
        descibe = pd.DataFrame(
            {
                "count": usfs.count_(selected_types),
                "mean": usfs.mean_(selected_types),
                "std": usfs.std_(selected_types),
                "min": usfs.min_(selected_types),
                "25%": usfs.quantile_(selected_types, 0.25),
                "50%": usfs.quantile_(selected_types, 0.50),
                "75%": usfs.quantile_(selected_types, 0.75),
                "max": usfs.max_(selected_types),
                "cv": usfs.cv_(selected_types),
                "range": usfs.range_(selected_types),
            }
        )
        print(descibe.T)
        # print("____________________") # test
        # print(selected_types.describe())
    except Exception as e:
        print("exception:", str(e))


if __name__ == "__main__":
    main()
