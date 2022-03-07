import json

# collating homelessness related data from squad
with open("../data/squad/train-v2.0.json", "r") as file:
    squad = json.loads(file.read())

QA = []

squad_data = squad["data"]
for d in squad_data:
    paragraphs = d["paragraphs"]
    for p in paragraphs:
        context = p["context"]
        if "homeless" in context or "transitional housing" in context or "affordable housing" in context:
            for question in p["qas"]:
                if len(question["answers"]) > 0:
                    entry = {
                        "context": context,
                        "question": question["question"],
                        "answers": question["answers"][0]["text"]
                    }
                    QA.append(entry)

# collating homelessness related data from hot pot data
with open("../data/squad/hotpot_train_v1.1.json", "r") as file:
    hotpot = json.loads(file.read())

for d in hotpot:
    paragraphs = d["context"]
    for p in paragraphs:
        context = p[1]
        for c in context:
            if "homeless" in c or "transitional housing" in c or "affordable housing" in c:
                # text
                text = context[0]
                for entry in context[1:]:
                    text += " " + entry

                # data
                entry = {
                    "context": text,
                    "question": d["question"],
                    "answers": d["answer"]
                }
                QA.append(entry)

with open("../data/qa.json", "w") as file:
    file.write(json.dumps(QA))



