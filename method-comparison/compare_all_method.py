import pandas as pd
from pathlib import Path

CATE = "all"

FILE_TAGS = ["keybert", "rake", "textrank", "yake"]
FILE_LOCS = ["./keybert-2020/predict/test_original.csv","./rake-2010/predict/test.csv", "./textrank-2004/predict/test.csv","./yake-2020/predict/test.csv"]

if __name__ == '__main__':
    df1 = pd.read_csv(FILE_LOCS[0]).astype(str)
    df1.drop(df1.columns[0], inplace=True, axis=1)
    df1.rename(columns={df1.columns[2]: FILE_TAGS[0]}, inplace=True)


    for i in range (1, len(FILE_TAGS)):
        file_loc, file_tag = FILE_LOCS[i], FILE_TAGS[i]
        df2 = pd.read_csv(file_loc).astype(str)
        df1[file_tag] = df2[df2.columns[3]]

    Path("./compare_result").mkdir(parents=True, exist_ok=True)
    df1.to_csv(f"./compare_result/{CATE}.csv")
    print("done!")
