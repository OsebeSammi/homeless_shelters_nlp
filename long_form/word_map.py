import json
from transformers import AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("roberta-large")
wikitext = load_dataset("wikitext", "wikitext-2-v1")
squad = load_dataset("squad_v2")

word_file = "../data/wordNetJson/wn.json"
with open(word_file, "r") as file:
    word_net = json.loads(file.read())
word_map = {}
ignore = "in by on it"


def process_wiki(wiki):
    for i in range(len(wiki["text"])):
        text = wiki["text"][i]
        words = text.split(" ")
        if len(words) > 20:
            for word in words:
                word = str.lower(word.strip())
                if word in word_net["words"] and word not in word_map and word not in ignore:
                    keys = word_net["words"][word]
                    if len(keys) > 0:
                        related_words = word_net["synsets"][keys[0]]
                        if "synonyms" in related_words:
                            synonymns = related_words["synonyms"]
                            synonym = str.lower(synonymns[0].strip())
                            if word != synonym and synonym not in ignore:
                                word_map[word] = synonym
                                word_map[synonym] = word


def process_squad(squad):
    questions = [q.strip() for q in squad["question"]]
    context = squad["context"]
    for i in range(len(questions)):
        text = questions[i] + " " + context[i].strip()
        words = text.split(" ")
        for word in words:
            word = str.lower(word.strip())
            if word in word_net["words"] and word not in word_map and word not in ignore:
                keys = word_net["words"][word]
                if len(keys) > 0:
                    related_words = word_net["synsets"][keys[0]]
                    if "synonyms" in related_words:
                        synonymns = related_words["synonyms"]
                        synonym = str.lower(synonymns[0].strip())
                        if word != synonym:
                            word_map[word] = synonym
                            word_map[synonym] = word


# wikitext["train"].map(process_wiki, batched=True, remove_columns=wikitext["train"].column_names)
squad["train"].map(process_squad, batched=True, remove_columns=squad["train"].column_names)
squad["validation"].map(process_squad, batched=True, remove_columns=squad["validation"].column_names)
# squad test
with open("squad_dev.json", "r") as file:
    squad_dev = json.loads(file.read())
squad_dev = squad_dev["data"]

questions = []
contexts = []
for i in range(len(squad_dev)):
    for j in range(len(squad_dev[i]["paragraphs"])):
        for k in range(len(squad_dev[i]["paragraphs"][j]["qas"])):
            question = squad_dev[i]["paragraphs"][j]["qas"][k]["question"]
            context = squad_dev[i]["paragraphs"][j]["context"]
            questions.append(question)
            contexts.append(context)
process_squad({"question": questions, "context": contexts})

with open("squad_synonyms.json", "w") as file:
    print(len(word_map), "pairs")
    file.write(json.dumps(word_map))

