import time

text_pth = "train.csv"
max_n = 6
min_freq = 3

# count time between start and end
def count_time(start, end):
    return round(end - start, 2)

def get_ngrams(sentence, n):
    """
    Split sentence into ngrams
    """
    ngrams = []
    if len(sentence) < n:
        return ngrams
    for i in range(len(sentence) - n + 1):
        ngrams.append(sentence[i:i+n])
    return ngrams

def extract_ngrams_py(text_pth, max_n, min_freq):
    """
    Split by newline and split into sentences, then get ngrams into dict and filter by min_freq
    """
    with open(text_pth, "r") as f:
        text = f.read()
    paragraphs = text.split("\n")
    sentences = []
    cut_signals = ["，","！","？","｡","。","\t","＂","＃","＄","％","＆","＇","（","）","＊","＋","，","－","／","：","；","＜","＝","＞","＠","［","＼","］","＾","＿","｀","｛","｜","｝","～","｟","｠","｢","｣","､","、","〃","》","「","」","『","』","【","】","〔","〕","〖","〗","〘","〙","〚","〛","〜","〝","〞","〟","〰","〾","〿","–","—","‘","'","‛","“","”","„","‟","…","‧","﹏","."]
    for paragraph in paragraphs:
        sentence = ""
        for c in paragraph:
            if c in cut_signals:
                sentences.append(sentence)
                sentence = ""
            else:
                sentence += c
        sentences.append(sentence)

    ngrams = {}
    for sentence in sentences:
        for n in range(1, max_n + 1):
            for ngram in get_ngrams(sentence, n):
                if ngram in ngrams:
                    ngrams[ngram] += 1
                else:
                    ngrams[ngram] = 1
    return {k: v for k, v in ngrams.items() if v >= min_freq}

# using python to get the frequency of ngrams
start = time.time()
results_py = extract_ngrams_py(text_pth, max_n, min_freq)
end = time.time()
print("Time Python:", count_time(start, end))