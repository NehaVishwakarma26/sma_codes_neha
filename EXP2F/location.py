import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("twitter_dataset.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())

countries=['India', 'USA', 'UK', 'Australia', 'Sri Lanka']
weights=[0.3,0.2,0.2,0.1,0.2]

df['location']=np.random.choice(countries,size=len(df),p=weights)

df['location']=df['location'].astype(str).str.strip().str.lower()

df=df[df['location']!='nan']
df=df[df['location']!='']

location_counts=df['location'].value_counts()

#bar chart
plt.figure()
location_counts.head(10).plot(kind='bar')
plt.show()

#pie chart
plt.figure()
location_counts.head(5).plot(kind='pie',autopct="%1.1f%%")
plt.show()

df['engagement']=df['Likes']+df['Retweets']
engagement_loc=df.groupby('location')['engagement'].sum().sort_values(ascending=False)
plt.figure()
engagement_loc.plot(kind='bar')
plt.show()

