import pandas as pd
import json


def get_annotated_data(dfs):
    eligibility = {
        "answers": [],
        "questions": [],
        "contexts": []
    }

    documents = {
        "answers": [],
        "questions": [],
        "contexts": []
    }

    services = {
        "answers": [],
        "questions": [],
        "contexts": []
    }

    admission = {
        "answers": [],
        "questions": [],
        "contexts": []
    }

    duration = {
        "answers": [],
        "questions": [],
        "contexts": []
    }

    for df in dfs:
        for i in range(len(df)-1):
            if len(df.iloc[i]["Eligibility & Requirements"].strip()) > 0:
                question = "What are the Eligibility & Requirement criteria?"
                eligibility["questions"].append(question)
                ans = df.iloc[i]["Eligibility & Requirements"]
                index = df.iloc[i]["Text"].find(ans)
                eligibility["answers"].append({
                    "answer_start": [index],
                    "text": [ans]
                })
                eligibility["contexts"].append(df.iloc[i]["Text"])

            if len(df.iloc[i]["Documents"].strip()) > 0:
                question = "What are the required documents?"
                documents["questions"].append(question)
                ans = df.iloc[i]["Documents"]
                index = df.iloc[i]["Text"].find(ans)
                documents["answers"].append({
                    "answer_start": [index],
                    "text": [ans]
                })
                documents["contexts"].append(df.iloc[i]["Text"])

            if len(df.iloc[i]["Services"].strip()) > 0:
                question = "What are the provided services?"
                services["questions"].append(question)
                ans = df.iloc[i]["Services"]
                index = df.iloc[i]["Text"].find(ans)
                services["answers"].append({
                    "answer_start": [index],
                    "text": [ans]
                })

                services["contexts"].append(df.iloc[i]["Text"])

            if len(df.iloc[i]["Admission Process"].strip()) > 0:
                question = "What is the admission process?"
                admission["questions"].append(question)
                ans = df.iloc[i]["Admission Process"]
                index = df.iloc[i]["Text"].find(ans)
                admission["answers"].append({
                    "answer_start": [index],
                    "text": [ans]
                })

                admission["contexts"].append(df.iloc[i]["Text"])

            if len(df.iloc[i]["Duration of Stay"].strip()) > 0:
                question = "How long can one stay?"
                duration["questions"].append(question)
                ans = df.iloc[i]["Duration of Stay"]
                index = df.iloc[i]["Text"].find(ans)
                duration["answers"].append({
                    "answer_start": [index],
                    "text": [ans]
                })

                duration["contexts"].append(df.iloc[i]["Text"])

    return eligibility, documents, services, admission, duration


def pack_few_shot(eligibility, documents, services, admission, duration, count):
    answers = []
    questions = []
    contexts = []
    for i in range(count):
        # eligibility
        answers.append(eligibility["answers"][i])
        questions.append(eligibility["questions"][i])
        contexts.append(eligibility["contexts"][i])

        # services
        answers.append(services["answers"][i])
        questions.append(services["questions"][i])
        contexts.append(services["contexts"][i])

        # admission
        answers.append(admission["answers"][i])
        questions.append(admission["questions"][i])
        contexts.append(admission["contexts"][i])

        # duration
        answers.append(duration["answers"][i])
        questions.append(duration["questions"][i])
        contexts.append(duration["contexts"][i])

        if i > len(documents):
            laps = int(count/len(duration))
            index = count - len(duration) * laps
        else:
            index = i

        # documents
        answers.append(documents["answers"][index])
        questions.append(documents["questions"][index])
        contexts.append(documents["contexts"][index])

    return answers, questions, contexts


def qa_bert_type(nlp, file_name):

    df = pd.read_excel("../../data/annotated/gold.xlsx", keep_default_na=False)
    validate_set = df[:50]

    # csv
    gold = []
    pred = {}

    questions = [
        "What are the Eligibility & Requirement criteria?",
        "What are the required documents?",
        "What are the provided services?",
        "What is the admission process?",
        "How long can one stay?"
    ]

    columns = ["Eligibility & Requirements", "Documents", "Services", "Admission Process", "Duration of Stay"]

    for i in range(len(validate_set)):
        index = str(validate_set.iloc[i]["Index"])
        context = validate_set.iloc[i]["Text"]
        paragraph = {
            "context": context,
            "qas": []
        }

        for j in range(len(questions)):
            if j == j: # line to be used to select label
                # model query
                qa = {
                    "question": questions[j],
                    "context": context
                }
                answer = nlp(qa)
                pred[index + "_" + str(j)] = answer["answer"]

                # print(index + "_" + str(j), answer["answer"])
                # print(validate_set.iloc[i][columns[j]], "\n")

                # gold
                if len(validate_set.iloc[i][columns[j]].strip()) == 0:
                    paragraph["qas"].append({
                        "question": questions[j],
                        "id": index + "_" + str(j),
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
    with open("../output/gold.json", "w") as file:
        file.writelines(json.dumps(gold))

    with open("../output/"+file_name+"_pred.json", "w") as file:
        file.writelines(json.dumps(pred))


def annotate(dfs, mask_key_words=False):
    answers = []
    questions = []
    contexts = []
    for df in dfs:
        for i in range(len(df)-1):
            if len(df.iloc[i]["Eligibility & Requirements"].strip()) > 0:
                question = "What are the Eligibility & Requirement criteria?"
                questions.append(question)
                ans = df.iloc[i]["Eligibility & Requirements"]
                index = df.iloc[i]["Text"].find(ans)
                answers.append({
                    "answer_start": [index],
                    "text": [ans]
                })

                contexts.append(df.iloc[i]["Text"])

            if len(df.iloc[i]["Documents"].strip()) > 0:
                question = "What are the required documents?"
                questions.append(question)
                ans = df.iloc[i]["Documents"]
                index = df.iloc[i]["Text"].find(ans)
                answers.append({
                    "answer_start": [index],
                    "text": [ans]
                })

                contexts.append(df.iloc[i]["Text"])

            if len(df.iloc[i]["Services"].strip()) > 0:
                question = "What are the provided services?"
                questions.append(question)
                ans = df.iloc[i]["Services"]
                index = df.iloc[i]["Text"].find(ans)
                answers.append({
                    "answer_start": [index],
                    "text": [ans]
                })

                contexts.append(df.iloc[i]["Text"])

            if len(df.iloc[i]["Admission Process"].strip()) > 0:
                question = "What is the admission process?"
                questions.append(question)
                ans = df.iloc[i]["Admission Process"]
                index = df.iloc[i]["Text"].find(ans)
                answers.append({
                    "answer_start": [index],
                    "text": [ans]
                })

                contexts.append(df.iloc[i]["Text"])

            if len(df.iloc[i]["Duration of Stay"].strip()) > 0:
                question = "How long can one stay?"
                questions.append(question)
                ans = df.iloc[i]["Duration of Stay"]
                index = df.iloc[i]["Text"].find(ans)
                answers.append({
                    "answer_start": [index],
                    "text": [ans]
                })

                contexts.append(df.iloc[i]["Text"])

    return answers, questions, contexts
