# from utils.csvt import CSVAnalyzer
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.sfs import max_, min_, mean_, std_, quantile_, count_, variance_
import sys
import pandas as pd

def main():

    if len(sys.argv) != 2: return

    df = pd.read_csv(sys.argv[1])
    selected_types = df.select_dtypes(include='number')
    descibe = pd.DataFrame({
        'count': count_(selected_types),
        'mean':mean_(selected_types),
        'std':std_(selected_types),
        'min':min_(selected_types),
        '25%':quantile_(selected_types, 0.25),
        '50%':quantile_(selected_types, 0.50),
        '75%':quantile_(selected_types, 0.75),
        'max':max_(selected_types),
        'var':variance_(selected_types),
    })
    print(descibe.T)
    # print("____________________") # test
    # print(selected_types.describe())
if __name__ == "__main__":
    main()