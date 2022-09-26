import json
from transformers import AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("roberta-large")
wikitext = load_dataset("wikitext", "wikitext-2-v1")

word_file = "../data/wordNetJson/wn.json"
with open(word_file, "r") as file:
    word_net = json.loads(file.read())
word_map = {}


def process(wiki):
    for i in range(len(wiki["text"])):
        text = wiki["text"][i]
        words = text.split(" ")
        if len(words) > 20:
            for word in words:
                word = str.lower(word.strip())
                if word in word_net["words"] and word not in word_map:
                    keys = word_net["words"][word]
                    if len(keys) > 0:
                        related_words = word_net["synsets"][keys[0]]
                        if "synonyms" in related_words:
                            synonymns = related_words["synonyms"]
                            synonym = str.lower(synonymns[0].strip())
                            if word != synonym:
                                word_map[word] = synonym
                                word_map[synonym] = word


wikitext["train"].map(process, batched=True, remove_columns=wikitext["train"].column_names)

with open("token_map.json", "w") as file:
    file.write(json.dumps(word_map))

