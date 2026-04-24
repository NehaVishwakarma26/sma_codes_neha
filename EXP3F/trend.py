import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("../EXP2F/twitter_dataset.csv")

print(df.head())

df['Timestamp']=pd.to_datetime(df['Timestamp'])

df['date']=df['Timestamp'].dt.date

trend=df.groupby('date').size()

plt.figure()
trend.head().plot(kind='bar')
plt.show()

plt.figure()
trend.plot(kind='line',color='blue')
plt.show()

df['engagement']=df['Likes']+df['Retweets']
eng_trend=df.groupby('date')['engagement'].sum()

plt.figure()
eng_trend.plot(kind='area',color='orange')
plt.show()

location_trends=df.groupby(['date','location']).size().unstack().fillna(0)
plt.figure()
location_trends.plot(kind='line',ax=plt.gca())
plt.show()

print(trend)