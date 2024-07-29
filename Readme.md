### GitHub Project Description

#### Word Embeddings Visualization Using Gensim and t-SNE

This project demonstrates how to create and visualize word embeddings using Gensim's `Word2Vec` model and dimensionality reduction with t-SNE. The following Python script trains a Word2Vec model on a set of sample sentences, reduces the dimensionality of the embeddings to 2D, and visualizes them using Matplotlib. Additionally, it prints the raw embeddings for each word.

#### Key Components:

1. **Data Preparation**:
   - The `data` variable contains sample sentences, where each sentence is a list of words. This is used to train the Word2Vec model.

2. **Model Training**:
   - `Word2Vec` from Gensim is used to train a model with the sample data. Parameters include:
     - `vector_size`: Dimension of the embedding vectors.
     - `window`: The maximum distance between the current and predicted words.
     - `min_count`: Ignores all words with a total frequency lower than this.
     - `workers`: Number of threads to use during training.

3. **Dimensionality Reduction**:
   - `TSNE` (t-distributed Stochastic Neighbor Embedding) is used to reduce the high-dimensional word embeddings to 2D for visualization. This helps in visualizing the similarity and distribution of words in a 2D space.

4. **Visualization**:
   - `Matplotlib` is used to create a scatter plot of the 2D embeddings. Each word is plotted, and its position is labeled for clarity.

5. **Raw Embeddings**:
   - The raw embeddings for each word are printed to the console. This shows the high-dimensional vector representation of each word.

