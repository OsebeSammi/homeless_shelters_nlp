import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("roberta-large")

word_file = "/Users/samosebe/Downloads/wordNetJson/wn.json"
with open(word_file, "r") as file:
    word_net = json.loads(file.read())

words = word_net["words"]
synonyms = word_net["synsets"]

word_map = {}
for word in words:
    key = words[word][0]
    if "synonyms" in synonyms[key]:
        syn = synonyms[key]["synonyms"]

        w = key
        w_token = max(tokenizer.encode(w))
        s = syn[0]
        s_token = max(tokenizer.encode(s))
        if w_token != s_token:
            word_map[w_token] = s_token
            word_map[s_token] = w_token
        print(w, w_token, s, s_token)

with open("token_map", "w") as file:
    file.write(json.dumps(word_map))

