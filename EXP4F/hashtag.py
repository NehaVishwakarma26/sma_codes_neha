import pandas as pd
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud

df=pd.read_csv('train.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
df=df.drop(columns=["keyword",'target','location'],axis=1)

df=df[df['text'].str.contains("#")]
print(df.head())

df['hashtags']=df['text'].str.findall(r'#\w+')
print(df['hashtags'].head(20))

all_hashtags=df['hashtags'].explode()
print(all_hashtags)

all_hashtags=all_hashtags.str.lower()

hashtag_counts=all_hashtags.value_counts()
print(hashtag_counts.head(5))

plt.figure()
hashtag_counts.head(10).plot(kind='barh')
plt.gca().invert_yaxis()
plt.show()

plt.figure()
hashtag_dict=hashtag_counts.to_dict()
wordcloud=WordCloud(background_color='white').generate_from_frequencies(hashtag_dict)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

plt.figure()
hashtag_counts.head(10).plot(kind='pie',autopct='%1.1f%%')
plt.show()