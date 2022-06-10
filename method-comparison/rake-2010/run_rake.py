###
# WARN: don't show probability

# import sys

# sys.path.extend(["./RAKE_tutorial"])

from rake import Rake
import jieba
import pandas as pd
from pathlib import Path
import re

STOPWORD_FILE = "../spacy_stopwords/zh.txt"
TEST_FILE = "../data/test.csv"

def get_stopwords(file_loc):
    stopwords = [word.lower().split('\n')[0] for word in open(STOPWORD_FILE, 'r', encoding='UTF-8')]
    return stopwords


if __name__ == '__main__':
    stopwords = get_stopwords(STOPWORD_FILE)
    reg = re.compile("[/\n]")
    rake = Rake(stopwords, max_words_length=1)

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

        # rake: take keyword after tokenization
        text = " ".join(list(jieba.cut_for_search(text)))
        keywords = rake.run(text)

        predict["keywords"].append([keyword[0] for keyword in keywords])

    df_predict = pd.DataFrame(predict)
    Path("./predict").mkdir(parents=True, exist_ok=True)
    df_predict.to_csv("./predict/test.csv")

    print(f"Total: {len(df.index)}")
    print("succeed!")