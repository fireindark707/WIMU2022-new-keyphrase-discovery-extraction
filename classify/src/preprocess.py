from utils import deduplicate,load_file_by_line,split_sentences
import pandas as pd
from collections import Counter

SENTENCE_MAX_LENGTH = 100
DATA_PATH = "/home/phoenix000/IT/new_word_discovery/data/"
WORD_TRAIN_PATH = "/home/phoenix000/IT/new_word_discovery/annotated_data/train_annotated.txt"
WORD_TEST_PATH = "/home/phoenix000/IT/new_word_discovery/annotated_data/test_annotated.txt"
DOC_TRAIN_PATH = "/home/phoenix000/IT/new_word_discovery/annotated_data/docs/train.csv"
DOC_TEST_PATH = "/home/phoenix000/IT/new_word_discovery/annotated_data/docs/test.csv"

def load_words_labels(filename:str) -> tuple:
    """
    Function to load the words from the file.
    """
    word_lines = load_file_by_line(filename)
    words = []
    labels = []
    for word_line in word_lines:
        word, label = word_line.split(" ")
        words.append(word)
        labels.append(label)
    return words, labels

def load_docs(filepath:str) -> list:
    """
    Function to load the documents from the file.
    """
    df = pd.read_csv(filepath)
    try:
        contents = df["content"].tolist()
        titles = df["title"].tolist()
    except KeyError:
        raise Exception("The file does not have the required columns.")
    docs = contents + titles
    return docs

def docs_to_sentences(docs:list) -> list:
    """
    Function to convert the documents to sentences.
    """
    sentences = []
    for doc in docs:
        sentences += split_sentences(doc)
    return sentences

def get_word_sentence_pairs(words_list:list, docs:list, labels_list:list) -> list:
    """
    Function to get the word-sentence pairs from the list of words.
    """
    word2label = dict(zip(words_list, labels_list))
    words_list = set(words_list)
    sentences = docs_to_sentences(docs)
    word_sentence_pairs = []
    for sen in sentences:
        if len(sen) > SENTENCE_MAX_LENGTH:
            continue
        for word in words_list:
            if word in sen and len(sen) > len(word):
                word_sentence_pairs.append((word, sen, word2label[word]))
    word_sentence_pairs = deduplicate(word_sentence_pairs)
    return word_sentence_pairs

def generate_data(mode="train") -> list:
    """
    Function to generate the training data.
    """
    if mode == "train":
        word_path = WORD_TRAIN_PATH
        doc_path = DOC_TRAIN_PATH
    elif mode == "test":
        word_path = WORD_TEST_PATH
        doc_path = DOC_TEST_PATH
    else:
        raise Exception("Invalid mode.")

    words, labels = load_words_labels(word_path)
    word_sentence_pairs = get_word_sentence_pairs(words, load_docs(doc_path), labels)

    labels = [pair[2] for pair in word_sentence_pairs]
    print(f"{len(word_sentence_pairs)} word-sentence pairs generated: {Counter(labels)}")
    text_lens = [len(pair[1]) for pair in word_sentence_pairs]
    print(f"The average text length is {sum(text_lens)/len(text_lens)}. Max is {max(text_lens)}. Minimum is {min(text_lens)}.")
    return word_sentence_pairs

def save_data(pairs,save_pth):
    """
    Function to save the data.
    """
    with open(DATA_PATH+save_pth, "w") as f:
        for pair in pairs:
            f.write(f"{pair[0]}\t{pair[1]}\t{pair[2]}\n")

if __name__ == '__main__':

    train_pairs = generate_data("train")
    test_pairs = generate_data("test")
    save_data(train_pairs,"train_data.txt")
    save_data(test_pairs,"test_data.txt")