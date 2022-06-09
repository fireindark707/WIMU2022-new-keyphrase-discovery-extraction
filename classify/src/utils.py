import re
from collections import Counter

def sent_tokenize(text):
    """
    Tokenizes a text into sentences. Using 。 ！ ？ as sentence delimiters.
    """
    text = re.sub(r"\s+", " ", text)
    return re.split(r"([。！？])", text)
    
def deduplicate(lst):
    """
    Removes duplicates from a list.
    """
    return list(set(lst))

def load_file_by_line(filename):
    """
    Loads a file line by line.
    """
    with open(filename, "r") as f:
        lines = f.readlines()
    return [l.strip() for l in lines]

def split_sentences(text):
    """
    Splits a text into sentences.
    """
    try:
        text = text.split("\n")
    except AttributeError:
        print("【Warning】The input is not a string:", text)
        return []
    sents = []
    for line in text:
        sents += sent_tokenize(line)
    return sents

def most_common_element(lst):
    """
    Returns the most common element in a list.
    """
    counter = Counter(lst)
    return counter.most_common(1)[0][0]

