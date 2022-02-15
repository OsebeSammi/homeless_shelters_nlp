import numpy as np
from sklearn.manifold import SpectralEmbedding, TSNE
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import os

model = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")

# load sentences
sentences = np.load("sentences.npy")

# get sentence embedding
sentence_embeddings = []
if os.path.exists("sentence_embeddings.npy"):
    sentence_embeddings = np.load("sentence_embeddings.npy")
else:
    for i in range(len(sentences)):
        print(i)
        sentence = sentences[i]
        embeddings = model.encode(sentence)
        sentence_embeddings.append(embeddings)

    # save sentence embeddings
    np.save("sentence_embeddings.npy", sentence_embeddings)

# manifold
# perplexity = np.arange(40, 50)
# print("Manifold")
s = sentence_embeddings[0::10]
# for p in perplexity:
p = 40
manifold = TSNE(n_components=2, init="random", random_state=0, perplexity=p, learning_rate='auto')
dimension_reduced = manifold.fit_transform(s)
print(p, "KL", manifold.kl_divergence_, "Features", manifold.n_features_in_, "iterations", manifold.n_iter_)
# plot
plt.scatter(dimension_reduced[:, 0], dimension_reduced[:, 1])
plt.savefig(str(p)+".png")
plt.close()

# save reduced dimension
np.save("reduced_dime.npy", dimension_reduced)




