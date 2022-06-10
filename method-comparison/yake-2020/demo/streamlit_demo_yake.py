import streamlit as st
import yake
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from spacy import displacy
import re
import jieba
import pandas as pd

TEST_FILE = "../../data/test.csv"
STOPWORD_FILE = "../../spacy_stopwords/zh.txt"

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
HTML_WRAPPER = """<div style="overflow-x: hidden; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""


st.sidebar.title("Yet Another Keyword Extractor (YAKE)")
st.sidebar.markdown("""
Unsupervised Approach for Automatic Keyword Extraction using Text Features.
https://liaad.github.io/yake/
"""
)

st.sidebar.markdown("""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LIAAD/yake/blob/gh-pages/notebooks/YAKE_tutorial.ipynb)
""")

st.sidebar.header('Parameters')

#side bar parameters
max_ngram_size = st.sidebar.slider("Select max ngram size", 1, 10, 1)
deduplication_thresold = st.sidebar.slider("Select deduplication threshold", 0.5, 1.0, 0.9)
numOfKeywords = st.sidebar.slider("Select number of keywords to return", 1, 50, 10)
deduplication_algo = st.sidebar.selectbox('deduplication function', ('leve','jaro','seqm'),2)


df = pd.read_csv(TEST_FILE).astype(str)
df.dropna()

input_text_demo = [row for row in df.index]


windowSize = 1

#User text in content
st.header('Demo')

selected_input_text = st.selectbox("Select sample text", input_text_demo)
text = st.text_area("Selected text", df[df.columns[1]][selected_input_text], 330)
language = "zh"

#use yake to extract keywords
stopwords = [word.lower().split('\n')[0] for word in open(STOPWORD_FILE, 'r', encoding='UTF-8')]

custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None, stopwords=stopwords)
cut_text = re.sub("[/\n]", " ", text)
cut_text = " ".join(list(jieba.cut_for_search(cut_text)))
keywords = custom_kw_extractor.extract_keywords(cut_text)




#get keywords and their position
ents = []
text_lower = text.lower()

keywords_list = str(keywords[0][0])
for m in re.finditer(keywords_list, text_lower):
    d = dict(start = m.start(), end = m.end(), label = "")
    ents.append(d)

for i in range(1, len(keywords)):
    kwords = str(keywords[i][0])
    keywords_list += (', ' + kwords)
    for m in re.finditer(kwords, text_lower):
        d = dict(start = m.start(), end = m.end(), label = "")
        ents.append(d)

#sort the result by ents, as ent rule suggests
sort_ents = sorted(ents, key=lambda x: x["start"])

st.header('Output')

result_view = st.radio("Choose visualization type",('Highlighting', 'Word cloud', 'Table'), index=0)
if result_view == 'Highlighting':
    #use spacy to higlight the keywords
    ex = [{"text": text,
        "ents": sort_ents,
        "title": None}]

    html = displacy.render(ex, style="ent", manual=True)
    html = html.replace("\n", " ")
    st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
elif result_view == "Table":
    #tabular data (columns: keywords, score)
    df = pd.DataFrame(keywords, columns=("keywords","score"))
    st.table(df)

else:
    #create and generate a word cloud image
    wordcloud = WordCloud(width = 1000, height = 600, max_font_size = 80,
                min_font_size=10, prefer_horizontal=1,
                max_words=numOfKeywords,
                background_color="white",
                collocations=False,
                regexp = r"\w[\w ']+").generate(keywords_list)

    #display the generated image
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)
