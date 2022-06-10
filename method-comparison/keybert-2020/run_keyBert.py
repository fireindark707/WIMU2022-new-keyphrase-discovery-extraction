from keybert import KeyBERT # https://github.com/MaartenGr/KeyBERT
import pandas as pd
from pathlib import Path
import re
import jieba

STOPWORD_FILE = "../spacy_stopwords/zh.txt"
TEST_FILE = "../data/test.csv"
reg = re.compile("[/\n]")

def get_stopwords(file_loc):
    stopwords = [word.lower().split('\n')[0] for word in open(STOPWORD_FILE, 'r', encoding='UTF-8')]
    return stopwords

def get_df_line(df):
    for row in df.index:
        yield tuple(re.sub(reg, " ", df[col][row]) for col in df.columns)

if __name__ == '__main__':
    stopwords = get_stopwords(STOPWORD_FILE)
    kb = KeyBERT() # default model: all-MiniLM-L6-v2

    df = pd.read_csv(TEST_FILE).astype(str)
    df.dropna()

    succ_counter = 0
    predict = {"title": [], "content": [], "keywords": []}
    for tp in get_df_line(df):
        predict["title"].append(tp[0])

        text = tp[1]
        predict["content"].append(text)

        # ketBert: take keyword after tokenization
        text = " ".join(list(jieba.cut_for_search(text)))
        keywords = kb.extract_keywords(text, stop_words=stopwords, top_n=10, )

        predict["keywords"].append([keyword[0] for keyword in keywords])

        succ_counter += 1

    df_predict = pd.DataFrame(predict)
    Path("./predict").mkdir(parents=True, exist_ok=True)
    df_predict.to_csv("./predict/test.csv")

    print(f"Total: {succ_counter}")
    print("succeed!")


