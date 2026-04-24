#---- sentiment analysis
import pandas as pd
from textblob import TextBlob
import re
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer

df=pd.read_csv('../EXP2F/twitter_dataset.csv')

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

def clean_text(text):
    text=text.lower()
    text=re.sub(r"[^a-zA-Z! ]"," ",text)
    return text

def get_polarity(text):
    return TextBlob(str(text)).sentiment.polarity

df['Text']=df['Text'].apply(clean_text)
df['polarity']=df['Text'].apply(get_polarity)

def classify_sentiment(score):
    if score>0:
        return "Positive"
    elif score<0:
        return "Negative"
    else:
        return "Neutral"

df['sentiment']=df['polarity'].apply(classify_sentiment)

#keyword extraction

vectorizer=CountVectorizer(stop_words='english',max_features=20)
matrix=vectorizer.fit_transform(df['Text'])
keywords=vectorizer.get_feature_names_out()
counts=matrix.sum(axis=0).A1
keyword_df=pd.DataFrame({'Keyword':keywords,'Count':counts})
keyword_df=keyword_df.sort_values(by='Count',ascending=False)
print(keyword_df.head(10))


#sentiment visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()
sns.countplot(data=df)
plt.show()

plt.figure()
sentiment_counts=df['sentiment'].value_counts()
sentiment_counts.plot(kind='pie')
plt.show()

plt.figure()
sns.histplot(df['polarity'])
plt.show()

plt.figure()
sns.boxplot(data=df,x='sentiment',y='Likes')
plt.show()