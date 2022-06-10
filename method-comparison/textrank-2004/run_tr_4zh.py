from textrank4zh import TextRank4Keyword, TextRank4Sentence # https://github.com/letiantian/TextRank4ZH
import pandas as pd
from pathlib import Path
import re
# import jieba
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
    # stopwords = get_stopwords(STOPWORD_FILE)

    df = pd.read_csv(TEST_FILE).astype(str)
    df.dropna()
    GDL = get_df_line(df)

    succ_counter = 0
    predict = {"title": [], "content": [], "keywords": []}
    # predict = {"title": [], "content": [], "keywords": [], "keyphrases": [], "summary": []}
    for i in tqdm.trange(len(df.index)):
        tp = next(GDL)
        tr4k = TextRank4Keyword(stop_words_file=STOPWORD_FILE)
        # tr4s = TextRank4Sentence(stop_words_file=STOPWORD_FILE)

        # title
        predict["title"].append(tp[0])

        # content
        text = tp[1]
        predict["content"].append(text)

        # keywords: take keyword after tokenization
        # keyphrase
        tr4k.analyze(text=text, lower=True, window=2)
        keywords = tr4k.get_keywords(10, word_min_len=1)
        # keyphrases = tr4k.get_keyphrases(keywords_num=10, min_occur_num=2)
        predict["keywords"].append([keyword.word for keyword in keywords]) # keyword.keys = ["word", "weight"]
        # predict["keyphrases"].append([keyphrase for keyphrase in keyphrases])

        # # summary
        # tr4s.analyze(text=text, lower=True, source = 'all_filters')
        # summaries = tr4s.get_key_sentences(num=1)
        # predict["summary"].append([summary.sentence for summary in summaries])
        # succ_counter += 1

    df_predict = pd.DataFrame(predict)
    Path("./predict").mkdir(parents=True, exist_ok=True)
    df_predict.to_csv("./predict/test_4zh.csv")

    print(f"Total: {succ_counter}")
    print("succeed!")


