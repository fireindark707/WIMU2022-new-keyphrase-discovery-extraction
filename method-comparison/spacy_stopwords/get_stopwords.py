import spacy

nlp = spacy.load("zh_core_web_sm")
spacy_stopwords = spacy.lang.zh.stop_words.STOP_WORDS

# print('spaCy has {} stop words'.format(len(spacy_stopwords)))
with open("./zh.txt",'w',encoding='UTF-8') as f:
    for word in spacy_stopwords:
        print(word, file=f)

# print(spacy_stopwords)