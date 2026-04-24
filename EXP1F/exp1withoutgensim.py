import pandas as pd
import re

from nltk.stem import PorterStemmer
from sentence_transformers.models.tokenizer import ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

df=pd.read_csv("bbc_news.csv")

print(df.head())

def clean_text(text):
    text=text.lower
    text=re.sub(r"[^a-zA-Z ]"," ",text)
    return text

stemmer=PorterStemmer()

def tokenizer(text):
    words=text.split()
    final_words=[]
    for word in words:
        if word not in ENGLISH_STOP_WORDS:
            stemmed=stemmer.stem(word)
            final_words.append(stemmed)

    return " ".join(final_words)

df['text']=df['title'].apply(clean_text)
df['processed']=df['text'].apply(tokenizer)

vectorizer=CountVectorizer(max_features=1000)

X=vectorizer.fit_transform(df['processed'])

lda=LatentDirichletAllocation(n_components=3,random_state=42)
lda.fit(X)

words=vectorizer.get_feature_names_out()

data=[]

for topic_idx,topic in enumerate(lda.components_):
    top_words_idx=topic.argsort()[-10:]
    for i in top_words_idx:
        data.append([topic_idx,words[i],topic[i]])

df_topics=pd.DataFrame(data,columns=["Topic","Word","Weight"])