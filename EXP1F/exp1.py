import pandas as pd
import re

from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import PorterStemmer

df=pd.read_csv("bbc_news.csv")

def clean_text(text):
    text=text.lower()
    text=re.sub(r"[^a-zA-Z# ]"," ",text)
    return text

stemmer=PorterStemmer()

def tokenize(text):
    words=text.split()
    final_words=[]
    for word in words:
        if word not in STOPWORDS:
            stemmed=stemmer.stem(word)
            final_words.append(stemmed)

    return final_words

df['text']=df['title']+" "+df['description']
df['text'] = df['text'].apply(clean_text)
df['tokens'] = df['text'].apply(tokenize)

#topic modelling
dictionary=corpora.Dictionary(df['tokens'])

corpus=[dictionary.doc2bow(text) for text in df['tokens']]

lda_model=LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=3,
    passes=3
)

topics=lda_model.show_topics(num_topics=3,num_words=10,formatted=False)

data=[]

for topic_no,words in topics:
    for word,weight in words:
        data.append([topic_no,word,weight])

df_topics=pd.DataFrame(data,columns=["Topic","Word","Weight"])


