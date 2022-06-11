from keybert import KeyBERT # https://github.com/MaartenGr/KeyBERT
import pandas as pd
from pathlib import Path
import re
import jieba

FILE_TYPE = "test"
STOPWORD_FILE = "../spacy_stopwords/zh.txt"

TEST_FILE = f"../data/{FILE_TYPE}.csv"
DICT_FILE = f"../data/tagged/{FILE_TYPE}_dict.txt"
OUT_FILE_PARENT = "./predict"
OUT_FILE = f"{OUT_FILE_PARENT}/{FILE_TYPE}_v8_modify_model_with_jieba_use_cut_for_search.csv"

reg = re.compile("[/\n]")

def get_stopwords(file_loc):
    stopwords = [word.lower().split('\n')[0] for word in open(file_loc, 'r', encoding='UTF-8')]
    return stopwords

def get_df_line(df):
    for row in df.index:
        yield tuple(re.sub(reg, " ", df[col][row]) for col in df.columns)

def get_my_dict(dict_loc):
    for word in open(dict_loc, "r", encoding='utf-8'):
        yield word.split('\n')[0] # now doesn't provide frequency & tag

if __name__ == '__main__':
    stopwords = get_stopwords(STOPWORD_FILE)
    kb = KeyBERT(model="paraphrase-multilingual-MiniLM-L12-v2") # default model: all-MiniLM-L6-v2
    # jieba initials
    for my_word in get_my_dict(DICT_FILE):
        jieba.add_word(my_word)
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
        text_tokens = jieba.cut(text)
        # text_tokens = jieba.cut_for_search(text)
        # text_tokens = set(list(jieba.cut_for_search(text)))

        # for token1 in text_tokens.copy():
        #     for token2 in text_tokens.copy():
        #         if len(token1) <= len(token2):
        #             continue
        #         elif token2 in token1:
        #             text_tokens.remove(token2)

        text = " ".join(list(text_tokens))
        keywords = kb.extract_keywords(text, stop_words=stopwords, top_n=10, diversity=0.2, use_mmr=True)
        # keywords = kb.extract_keywords(text, stop_words=stopwords, top_n=50, )

        # keywords.sort()
        # for k1 in keywords.copy():
        #     for k2 in keywords.copy():
        #         if len(k1[0]) <= len(k2[0]):
        #             continue
        #         elif k2[0] in k1[0]:
        #             keywords.remove(k2)

        # new_keywords = []
        # for keyword in keywords:
        #     new_keywords.append((keyword[1], keyword[0]))
        # new_keywords.sort(reverse=True)

        # predict["keywords"].append([keyword[1] for keyword in new_keywords[:10]])
        predict["keywords"].append([keyword[0] for keyword in keywords])

        succ_counter += 1

    df_predict = pd.DataFrame(predict)
    Path(OUT_FILE_PARENT).mkdir(parents=True, exist_ok=True)
    df_predict.to_csv(OUT_FILE)

    print(f"Total: {succ_counter}")
    print("succeed!")


