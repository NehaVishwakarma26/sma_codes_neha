import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

from EXP5.sentiment import vectorizer, keyword_df

try:
    df=pd.read_csv('../EXP6/Social Media Engagement Dataset.csv',encoding="utf-8")
except:
    df=pd.read_csv("../EXP6/Social Media Engagement Dataset.csv",encoding="latin-1")

print(df.info())
print(df['brand_name'].value_counts())

brand_df=df[df['brand_name']=='Apple'].copy()

#sentiment
def get_polarity(text):
    return TextBlob(text).sentiment.polarity

def get_sentiment(score):
    if score<0:
        return "Negative"
    elif score>0:
        return "Positive"
    else:
        return "Neutral"

brand_df['polarity']=brand_df['text_content'].apply(get_polarity)
brand_df['sentiment']=brand_df['polarity'].apply(get_sentiment)
sentiment=brand_df['sentiment'].value_counts()

#plot
plt.figure()
sentiment.plot(kind='pie',autopct='%1.1f%%')
plt.title("Sentiment distribution for Apple")
plt.show()

#keyword extraction

vectorizer=CountVectorizer(stop_words='english',max_features=20)
matrix=vectorizer.fit_transform(brand_df['text_content'])
keywords=vectorizer.get_feature_names_out()
counts=matrix.sum(axis=0).A1

keyword_df=pd.Dataframe({'keyword':keywords,'count':counts})
keyword_df=keyword_df.sort_values(by='count',ascending=False)

plt.figure()
