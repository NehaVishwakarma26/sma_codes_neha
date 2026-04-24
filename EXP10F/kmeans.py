import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Load data
df = pd.read_csv("../EXP6F/Social Media Engagement Dataset.csv")

# 2. Select features
features = df[['likes_count', 'comments_count', 'shares_count', 'sentiment_score']]

# 3. Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(features)

# 4. Cluster summary
summary = df.groupby('cluster')[features.columns].mean()
print(summary)

# -------------------------------
# 🔥 VISUAL 1: Scatter plot
# -------------------------------
plt.figure()
plt.scatter(df['likes_count'], df['comments_count'], c=df['cluster'])
plt.xlabel("Likes")
plt.ylabel("Comments")
plt.title("Clusters (Likes vs Comments)")
plt.show()

# -------------------------------
# 🔥 VISUAL 2: Cluster size (BAR)
# -------------------------------
sizes = df['cluster'].value_counts().sort_index()

plt.figure()
plt.bar(sizes.index, sizes.values)
plt.xlabel("Cluster")
plt.ylabel("Number of Posts")
plt.title("Cluster Sizes")
plt.show()

# -------------------------------
# 🔥 VISUAL 3: Box plot (ENGAGEMENT)
# -------------------------------
df['engagement'] = df['likes_count'] + df['comments_count'] + df['shares_count']

plt.figure()
df.boxplot(column='engagement', by='cluster')
plt.title("Engagement Distribution by Cluster")
plt.suptitle("")   # removes extra title
plt.show()

# -------------------------------
# 🔥 VISUAL 4: Pie chart
# -------------------------------
plt.figure()
plt.pie(sizes.values, labels=sizes.index, autopct='%1.1f%%')
plt.title("Cluster Distribution")
plt.show()