from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import requests

model = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")
clusters = np.load("reduced_dime.npy")

clusters_dims = {
    0: [0, 20, 10, -10],      # center of sphere
    1: [-59, -39, 30, 10],    # cluster 1
    2: [-39, -18, 40, 20],    # cluster 2
    3: [-20, 1, 60, 40],      # cluster 3
    4: [31, 42, 20, 4],       # cluster 4
    5: [48, 60, 12, -2],      # cluster 5
    6: [48, 60, -2, -20],     # cluster 6
    7: [-45, -20, -10, -28],  # cluster 7
    8: [2, 30, -33, -60],     # cluster 8
    9: [-18, 2, -20, -40],    # bottom left
    10: [-30, -10, 10, -18]   # left
}
cluster = 10
rectangle = clusters_dims[cluster]
cluster_name = "../plots/word_cloud_cluster_"+str(cluster)+".png"

# Sentences
sentences = np.load("sentences.npy")
sample_sentences = sentences[0::10]
clean = ""

for i in range(len(sample_sentences)):
    c = clusters[i]
    if rectangle[0] < c[0] < rectangle[1] and rectangle[2] > c[1] > rectangle[3]:
        s = sample_sentences[i]
        print(c)
        print(s)
        clean += s + " "

# cloud
pic = np.array(Image.open(requests.get(
    'http://www.clker.com/cliparts/2/9/b/8/1194984775760075334button-green_benji_park_01.svg.med.png',
    stream=True).raw))

# with collocations
wordcloud = WordCloud(width=1000, height=1000, background_color='white', mask=pic,
                      min_font_size=10, collocations=False).generate(clean)
plt.figure(figsize=(15, 15), facecolor='white', edgecolor='blue')
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig(cluster_name)
