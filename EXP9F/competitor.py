import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import  TextBlob

try:
    df=pd.read_csv("../EXP6F/Social Media Engagement Dataset.csv",encoding="utf-8")
except:
    df=pd.read_csv("../EXP6F/Social Media Engagement Dataset.csv",encoding="latin-1")

# competitor analaysis
#sentiment
#engagement
#no of posts
#heatmap for brand vs engagement

df['engagement']=0.5*df['likes_count']+0.3*df['comments_count']+0.2*df['shares_count']

def get_polarity(text):
    return TextBlob(text).sentiment.polarity

def get_sentiment(score):
    if score<0:
        return "Negative"
    elif score>0:
        return "Positive"
    else:
        return "Neutral"

df['polarity']=df['text_content'].apply(get_polarity)
df['sentiment']=df['polarity'].apply(get_sentiment)

sentiment_brand=df.groupby(['brand_name','sentiment']).size().unstack()

plt.figure()
sentiment_brand.plot(kind='bar',stacked=True)
plt.show()

engagement_brand=df.groupby(df['brand_name'])['engagement'].sum()

plt.figure()
df.boxplot(column='engagement',by='brand_name')
plt.show()

post_count=df['brand_name'].value_counts()

plt.figure()
post_count.plot(kind='pie')
plt.show()

#heatmap
heat = df.groupby(['brand_name','sentiment'])['engagement'].mean().unstack()
plt.figure()
sns.heatmap(heat)
plt.show()