from keybert import KeyBERT  # https://github.com/MaartenGr/KeyBERT
import pandas as pd
from pathlib import Path
import re
import jieba
import json


FILE_TYPE = "test"
STOPWORD_FILE = "../spacy_stopwords/zh.txt"
MY_KEYWORD_WEIGHT = 1.2
REGEX_PATTERN = "[/\n]"

TEST_FILE = f"../data/{FILE_TYPE}.csv"
DICT_FILE = f"../data/tagged/my_dict.json"
OUT_FILE_PARENT = "./predict"
OUT_FILE = f"{OUT_FILE_PARENT}/{FILE_TYPE}_our_strategy.csv"


def get_stopwords(file_loc):
    stopwords = [
        word.lower().split("\n")[0] for word in open(file_loc, "r", encoding="UTF-8")
    ]
    return stopwords


def get_df_line(df):
    for row in df.index:
        yield tuple(
            re.sub(REGEX_PATTERN, " ", df[col][row]) for col in df.columns
        )  # title content


def get_my_dict(dict_loc):
    for word in open(dict_loc, "r", encoding="utf-8"):
        yield word.split("\n")[0]  # now doesn't provide frequency & tag


if __name__ == "__main__":
    stopwords = get_stopwords(STOPWORD_FILE)
    # default model: all-MiniLM-L6-v2
    kb = KeyBERT(model="paraphrase-multilingual-MiniLM-L12-v2")

    # jieba initials
    weighted_dict = []
    with open(
        DICT_FILE,
        "r",
        encoding="utf-8",
        newline="",
    ) as jsonfile:
        data = json.load(jsonfile)

        for f in data:  # level_1 equals to "train" or "test" file
            for key in data[f]:  # level_2 equals to "O", "Name", "Labor"
                if (
                    key == "Labor"
                ):  # remove "O", "00", "Name" to enhance weight of "Labor"，
                    for my_word in data[f][key]:
                        jieba.add_word(my_word, 4)  # add weight to self-dict
                        weighted_dict.append(my_word)
                elif key not in {"O", "OO"}:  # "General", "Name"
                    for my_word in data[f][key]:
                        jieba.add_word(my_word, 4)  # add weight to self-dict
    print("jieba load dict done!")

    # deal with data
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
        keywords = kb.extract_keywords(
            text, stop_words=stopwords, top_n=50, diversity=0.2, use_mmr=True
        )

        new_keywords = []
        for keyword in keywords:
            if keyword[0] in weighted_dict:
                new_keywords.append((keyword[1] * MY_KEYWORD_WEIGHT, keyword[0]))
            else:
                new_keywords.append((keyword[1], keyword[0]))
        new_keywords.sort(reverse=True)
        predict["keywords"].append([keyword[1] for keyword in new_keywords[:10]])

        succ_counter += 1

    df_predict = pd.DataFrame(predict)
    Path(OUT_FILE_PARENT).mkdir(parents=True, exist_ok=True)
    df_predict.to_csv(OUT_FILE)

    print(f"Total: {succ_counter}")
    print("succeed!")
