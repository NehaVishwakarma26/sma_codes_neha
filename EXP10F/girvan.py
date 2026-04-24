import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from networkx.algorithms.community import girvan_newman

# 1. Load + sample
df = pd.read_csv("../EXP6F/Social Media Engagement Dataset.csv").sample(500)

# 2. Text → numbers
X = CountVectorizer(stop_words='english').fit_transform(df['text_content'])

# 3. LDA → topics
topics = LatentDirichletAllocation(n_components=5).fit_transform(X)
df['topic'] = topics.argmax(axis=1)

# 4. Build graph
G = nx.Graph()
for t, g in df.groupby('topic'):
    u = g['user_id'].unique()[:30]
    for i in range(len(u)):
        for j in range(i+1, len(u)):
            G.add_edge(u[i], u[j])

# 5. Girvan-Newman
comm = list(next(girvan_newman(G)))

# -------------------------------
# 🔥 VISUAL 1: Community size (BAR)
# -------------------------------
sizes = [len(c) for c in comm]
plt.figure()
plt.bar(range(len(sizes)), sizes)
plt.title("Community Sizes")
plt.show()

# -------------------------------
# 🔥 VISUAL 2: PIE chart
# -------------------------------
plt.figure()
plt.pie(sizes, autopct="%1.1f%%")
plt.title("Community Distribution")
plt.show()

# -------------------------------
# 🔥 VISUAL 3: Network graph (small)
# -------------------------------
sub_nodes = list(G.nodes())[:100]
subG = G.subgraph(sub_nodes)

plt.figure(figsize=(6,6))
nx.draw(subG, node_size=50, with_labels=False)
plt.title("Network Graph")
plt.show()

# -------------------------------
# 🔥 VISUAL 4: Influential users
# -------------------------------
cent = nx.degree_centrality(G)
top = sorted(cent.items(), key=lambda x: x[1], reverse=True)[:10]

users = [u for u,_ in top]
scores = [s for _,s in top]

plt.figure()
plt.bar(users, scores)
plt.xticks(rotation=45)
plt.title("Top Users")
plt.show()