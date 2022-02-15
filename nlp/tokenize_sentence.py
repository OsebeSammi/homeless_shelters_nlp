from nltk.tokenize import sent_tokenize
import json
import numpy as np

# load data
with open("../data/dataset.json", "r") as file:
    shelters = json.loads(file.read())

sentences = []
for key in shelters:
    print(key)
    shelter = shelters[key]
    descriptions = shelter["description"]
    for desc in descriptions:
        s = sent_tokenize(desc["description"])
        sentences = np.concatenate((sentences, s))

np.save("sentences.npy", sentences)

