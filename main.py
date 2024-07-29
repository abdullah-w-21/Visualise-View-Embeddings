import gensim
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

#data to convert to embedding
data = [
    ['this', 'is', 'a', 'big', 'house'],
    ['King', 'Queen', 'Man', 'Women', 'Human'],
    ['one', 'two', 'three'],
    ['apple', 'mango', 'oranges', 'strawberry'],
    ['car', 'tractor', 'motor', 'machine', 'animals']
]

#training small word2vec model
model_test = Word2Vec(data, vector_size=100, window=5, min_count=1, workers=4)

#getting raw embedding data
words = list(model_test.wv.index_to_key)
embeddings = np.array([model_test.wv[word] for word in words])  # Convert list to numpy array

#t-distributed Stochastic Neighbor Embedding(dimensionality reduction technique) reduce size of embedding so we can view raw data
tsne = TSNE(n_components=2, random_state=0, perplexity=min(5, len(words) - 1))
embeddings_2d = tsne.fit_transform(embeddings)

#plot
plt.figure(figsize=(12, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', edgecolors='k')

for word, (x, y) in zip(words, embeddings_2d):
    plt.text(x + 0.1, y + 0.1, word, fontsize=12)

plt.xlabel('TSNE Component 1')
plt.ylabel('TSNE Component 2')
plt.title('2D Visualization of Word Embeddings')
plt.show()

#raw embeds
for word in words:
    print(f'Word: {word}')
    print(f'Embedding: {model_test.wv[word]}\n')
