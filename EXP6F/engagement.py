import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("Social Media Engagement Dataset.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

df['engagement']=df['likes_count']+df['shares_count']+df['comments_count']

user_engagement=df.groupby('user_id')['engagement'].sum().sort_values(ascending=False)

print(user_engagement)

#top users
#plot
plt.figure()
user_engagement.head(10).plot(kind='barh')
plt.xlabel("Engagement")
plt.ylabel("User id")
plt.title("Top 10 users by engagement")
plt.gca().invert_yaxis()
plt.show()

top_posts=df.sort_values(by='engagement',ascending=False)

plt.figure()
plt.hist(df['engagement'])
plt.xlabel("Engagement")
plt.ylabel("Number of Posts")
plt.title("Distribution of Engagement")
plt.show()

df['timestamp']=pd.to_datetime(df['timestamp'])

dates=df['timestamp'].dt.date

trend=df.groupby(df['timestamp'].dt.date)['engagement'].sum()
plt.figure()
trend.plot(kind='line')
plt.xlabel("Date")
plt.ylabel("Total engagement")
plt.title("Engagement trend over time")
plt.show()

sentiment_engagement=df.groupby('sentiment_label')['engagement'].sum()
plt.figure()
sentiment_engagement.head(10).plot(kind='pie',autopct="%1.1f%%")
plt.show()