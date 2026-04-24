import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    df=pd.read_csv('../EXP6F/Social Media Engagement Dataset.csv',encoding="utf-8")
except:
    df=pd.read_csv("../EXP6F/Social Media Engagement Dataset.csv",encoding="latin-1")

print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df.describe())

print(df.head())

df['engagement']=df['likes_count']+df['comments_count']+df['shares_count']

#histogram
plt.figure()
plt.hist(df['engagement'])
plt.xlabel("Engagement")
plt.ylabel("Number of posts with engagement")
plt.title("Engagement vs Number of Posts")
plt.show()

#boxplot
plt.figure()
plt.boxplot([df['likes_count'], df['comments_count'], df['shares_count']])
plt.xticks([1,2,3], ['Likes','Comments','Shares'])
plt.title("Comparison of Engagement Components")
plt.show()

df.boxplot(column='engagement', by='platform')
plt.title("Engagement by Platform")
plt.suptitle("")  # removes default title
plt.show()

cols=[
    'likes_count',
    'comments_count',
    'shares_count',
    'impressions',
    'engagement'
]

corr=df[cols].corr()

plt.figure(figsize=(6,4))
sns.heatmap(corr,annot=True)
plt.title("Correlation heatmap")
plt.show()