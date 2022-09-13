from transformers import pipeline
import pandas as pd
import pickle
import evaluate as squad_eval

with open("models/roberta_cross_1.5e-5", "rb") as file:
    model = pickle.load(file)
nlp = pipeline('question-answering', model=model, tokenizer="deepset/roberta-base-squad2")

# model_name = "deepset/roberta-base-squad2"
# nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)


questions = [
    "What are the Eligibility & Requirement criteria?",
    "What are the required documents?",
    "What are the provided services?",
    "What is the admission process?",
    "How long can one stay?"
]

columns = ["Eligibility & Requirements", "Documents", "Services", "Admission Process", "Duration of Stay"]

# load dataset
# with open("../data/dataset.json", "r") as file:
#     dataset = json.loads(file.read())
df = pd.read_excel("../data/annotated/gold.xlsx", keep_default_na=False)
validate_set = df[:50]

# csv
gold = []
pred = {}
for i in range(len(validate_set)):
    index = str(validate_set.iloc[i]["Index"])
    context = validate_set.iloc[i]["Text"]
    paragraph = {
        "context": context,
        "qas": []
    }

    for j in range(len(questions)):
        if j == j:
            # model query
            qa = {
                "question": questions[j],
                "context": context
            }
            answer = nlp(qa)
            pred[index+"_"+str(j)] = answer["answer"]

            print(index+"_"+str(j), answer["answer"])
            print(validate_set.iloc[i][columns[j]], "\n")

            # gold
            if len(validate_set.iloc[i][columns[j]].strip()) == 0:
                paragraph["qas"].append({
                    "question": questions[j],
                    "id": index+"_"+str(j),
                    "answers": [],
                    "is_impossible": True
                })

            else:
                if "###" in validate_set.iloc[i][columns[j]]:
                    multi_answers = validate_set.iloc[i][columns[j]].split("###")
                    answers = []
                    for ans in multi_answers:
                        answers.append({
                                "text": ans,
                                "answer_start": context.find(ans)
                            })
                else:
                    answers = [{
                        "text": validate_set.iloc[i][columns[j]],
                        "answer_start": context.find(validate_set.iloc[i][columns[j]])
                    }]

                paragraph["qas"].append({
                    "question": questions[j],
                    "id": index + "_" + str(j),
                    "answers": answers,
                    "is_impossible": False
                })

    gold.append(paragraph)

# write
gold = {
    "data": [
        {
            "title": "Homelessness",
            "paragraphs": gold
        }
    ]
}

results = squad_eval.run(gold["data"], pred)
print(results)
