import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns


try:
    df=pd.read_csv('../EXP6F/Social Media Engagement Dataset.csv',encoding="utf-8")
except:
    df=pd.read_csv("../EXP6F/Social Media Engagement Dataset.csv",encoding="latin-1")

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

keyword_df=pd.DataFrame({'keyword':keywords,'count':counts})
keyword_df=keyword_df.sort_values(by='count',ascending=False)

plt.figure()
plt.bar(keyword_df['keyword'].head(10),keyword_df['count'].head(10))
plt.show()

brand_df['engagement']=(brand_df['likes_count']+brand_df['comments_count']+brand_df['shares_count'])
plt.figure()
plt.hist(brand_df['engagement'],bins=20)
plt.show()

#brand engagement over time
brand_df['timestamp']=pd.to_datetime(brand_df['timestamp'])
trend=brand_df.groupby(brand_df['timestamp'].dt.to_period('M'))['engagement'].sum()

plt.figure()
trend.plot(kind="line")
plt.show()