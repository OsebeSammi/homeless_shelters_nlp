import json
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import utils.utils as utilities
from wordcloud import WordCloud
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import requests

with open("../data/dataset.json", "r") as file:
    shelters = json.loads(file.read())

stops_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
clean = ""

# preprocess
for key in shelters:
    shelter = shelters[key]
    descriptions = shelter["description"]
    token_ids = []

    for desc in descriptions:
        # remove capitalization
        words = utilities.get_words(str.lower(desc["description"]))

        # remove stop words
        for word in words:
            # check
            if word not in stops_words:
                # stem words
                # word = stemmer.stem(word)
                clean += word + " "


# cloud
pic = np.array(Image.open(requests.get(
    'http://www.clker.com/cliparts/2/9/b/8/1194984775760075334button-green_benji_park_01.svg.med.png',
                                       stream=True).raw))

# with collocations
wordcloud = WordCloud(width=1000, height=1000, background_color='white', mask=pic,
                      min_font_size=10, collocations=True).generate(clean)
plt.figure(figsize=(15, 15), facecolor='white', edgecolor='blue')
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig("../plots/word_cloud_collocations.png")

# without collocations
wordcloud = WordCloud(width=1000, height=1000, background_color='white', mask=pic,
                      min_font_size=10, collocations=False).generate(clean)
plt.imshow(wordcloud)
plt.savefig("../plots/word_cloud.png")
