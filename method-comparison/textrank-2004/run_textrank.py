from summa import keywords # https://github.com/summanlp/textrank
import pandas as pd
from pathlib import Path
import re
import jieba
import tqdm

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

    df = pd.read_csv(TEST_FILE).astype(str)
    df.dropna()

    succ_counter = 0
    predict = {"title": [], "content": [], "keywords": []}
    GDL = get_df_line(df)
    for i in tqdm.trange(len(df.index)):
        tp = next(GDL)
        # title
        predict["title"].append(tp[0])
        # content
        text = tp[1]
        predict["content"].append(text)
        # keywords: take keyword after tokenization
        text = " ".join(list(jieba.cut_for_search(text)))
        tr_keywords = keywords.keywords(text, words=10, scores=True, additional_stopwords=stopwords) # I add zh_stopwords in additional_stopwords
        # tr_keywords = tr_keywords[:min(10, len(tr_keywords))]
        
        ## about keywords.keywords(text, ratio=0.2, words=None, language="english", split=False, scores=False, deaccent=False, additional_stopwords=None)
        # words: int, numbers of keywords, but sometimes it would not give 10 words
        #   use tr_keywords[:10] would perfectly return 10 words, but it cause accuracy lose
        # score: bool, return score at index[1] or not
        # additional_keywords: str_list, stopwords=set(language_stopwords + additional_stopwords)
        # split: bool, unknown

        predict["keywords"].append([tr_keyword[0] for tr_keyword in tr_keywords])

        succ_counter += 1

    df_predict = pd.DataFrame(predict)
    Path("./predict").mkdir(parents=True, exist_ok=True)
    df_predict.to_csv("./predict/test.csv")

    print(f"Total: {succ_counter}")
    print("succeed!")


