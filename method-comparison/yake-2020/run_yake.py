import yake # https://github.com/LIAAD/yake
import jieba
import pandas as pd
from pathlib import Path
import re

TEST_FILE = "../data/test.csv"
STOPWORD_FILE = "../spacy_stopwords/zh.txt"


if __name__ == "__main__":
    # config = Configure()
    reg = re.compile("[/\n]")
    stopwords = [word.lower().split('\n')[0] for word in open(STOPWORD_FILE, 'r', encoding='UTF-8')]

    # yake setting: can add self stopword
    kw_extractor = yake.KeywordExtractor(lan="zh", n=1, top=10, stopwords=stopwords)

    df = pd.read_csv(TEST_FILE).astype(str)
    df.dropna()

    predict = {"title": [], "content": [], "keywords": []}
    for row in df.index:
        # print(row)
        predict["title"].append(df[df.columns[0]][row])

        text = df[df.columns[1]][row]
        text = re.sub(reg, " ", text)
        # print(text)
        predict["content"].append(text)

        # yake: take keyword after tokenization
        text = " ".join(list(jieba.cut_for_search(text)))
        keywords = kw_extractor.extract_keywords(text)
        predict["keywords"].append([keyword[0] for keyword in keywords])

    df_predict = pd.DataFrame(predict)
    Path("./predict").mkdir(parents=True, exist_ok=True)
    df_predict.to_csv("./predict/test.csv")

    print(f"Total: {len(df.index)}")
    print("succeed!")
