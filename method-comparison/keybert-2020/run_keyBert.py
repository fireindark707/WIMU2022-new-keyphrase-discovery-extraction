from keybert import KeyBERT # https://github.com/MaartenGr/KeyBERT
import pandas as pd
from pathlib import Path
import re
import jieba

FILE_TYPE = "test"
STOPWORD_FILE = "../spacy_stopwords/zh.txt"

TEST_FILE = f"../data/{FILE_TYPE}.csv"
TRAIN_DICT_FILE = f"../data/tagged/train_dict.txt"
TEST_DICT_FILE = f"../data/tagged/test_dict.txt"
OUT_FILE_PARENT = "./predict"
OUT_FILE = f"{OUT_FILE_PARENT}/{FILE_TYPE}_v9_modify_v7_with_add_weight_to_my_dict.csv"

reg = re.compile("[/\n]")

def get_stopwords(file_loc):
    stopwords = [word.lower().split('\n')[0] for word in open(file_loc, 'r', encoding='UTF-8')]
    return stopwords

def get_df_line(df):
    for row in df.index:
        yield tuple(re.sub(reg, " ", df[col][row]) for col in df.columns) # title content

def get_my_dict(dict_loc):
    for word in open(dict_loc, "r", encoding='utf-8'):
        yield word.split('\n')[0] # now doesn't provide frequency & tag

if __name__ == '__main__':
    stopwords = get_stopwords(STOPWORD_FILE)
    kb = KeyBERT(model="paraphrase-multilingual-MiniLM-L12-v2") # default model: all-MiniLM-L6-v2
    # jieba initials
    for my_word in get_my_dict(TRAIN_DICT_FILE):
        jieba.add_word(my_word, 4)  # add weight to self-dict
    for my_word in get_my_dict(TEST_DICT_FILE):
        jieba.add_word(my_word, 4)  # add weight to self-dict
    print("jieba load dict done!")

    df = pd.read_csv(TEST_FILE).astype(str)
    df.dropna()

    succ_counter = 0
    predict = {"title": [], "content": [], "keywords": []}
    for tp in get_df_line(df):
        predict["title"].append(tp[0])

        text = tp[1]
        predict["content"].append(text)

        # ketBert: take keyword after tokenization
        text = " ".join(list(jieba.cut(text)))
        keywords = kb.extract_keywords(text, stop_words=stopwords, top_n=10, diversity=0.2, use_mmr=True)
        predict["keywords"].append([keyword[0] for keyword in keywords])

        succ_counter += 1

    df_predict = pd.DataFrame(predict)
    Path(OUT_FILE_PARENT).mkdir(parents=True, exist_ok=True)
    df_predict.to_csv(OUT_FILE)

    print(f"Total: {succ_counter}")
    print("succeed!")


