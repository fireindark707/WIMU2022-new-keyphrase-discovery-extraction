import pandas as pd
from pathlib import Path

CATE = "keybert"

ORIGINAL_FILE = "./keybert-2020/predict/test_original.csv"
OUR_FILE = "./keybert-2020/predict/test_our_strategy.csv"

if __name__ == '__main__':
    df1 = pd.read_csv(ORIGINAL_FILE).astype(str)
    df2 = pd.read_csv(OUR_FILE).astype(str)

    df1.drop(df1.columns[0], inplace=True, axis=1)
    df1['keyword_with_our_strategy'] = df2[df2.columns[3]]

    Path("./compare_result").mkdir(parents=True, exist_ok=True)
    df1.to_csv(f"./compare_result/{CATE}.csv")
    print("done!")
