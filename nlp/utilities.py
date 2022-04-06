def get_annotated_data(dfs):
    answers = []
    questions = []
    contexts = []
    for df in dfs:
        for i in range(len(df)-1):
            if len(df.iloc[i]["Eligibility & Requirements"]) > 0:
                question = "What are the Eligibility & Requirement criteria?"
                questions.append(question)
                ans = df.iloc[i]["Eligibility & Requirements"]
                index = df.iloc[i]["Text"].find(ans)
                answers.append({
                    "answer_start": [index],
                    "text": [ans]
                })
                contexts.append(mask(df.iloc[i]["Text"]))

            if len(df.iloc[i]["Documents"]) > 0:
                question = "What are the required documents?"
                questions.append(question)
                ans = df.iloc[i]["Documents"]
                index = df.iloc[i]["Text"].find(ans)
                answers.append({
                    "answer_start": [index],
                    "text": [ans]
                })
                contexts.append(mask(df.iloc[i]["Text"]))

            if len(df.iloc[i]["Services"]) > 0:
                question = "What are the provided services?"
                questions.append(question)
                ans = df.iloc[i]["Services"]
                index = df.iloc[i]["Text"].find(ans)
                answers.append({
                    "answer_start": [index],
                    "text": [ans]
                })
                contexts.append(mask(df.iloc[i]["Text"]))

            if len(df.iloc[i]["Admission Process"]) > 0:
                question = "What is the admission process?"
                questions.append(question)
                ans = df.iloc[i]["Admission Process"]
                index = df.iloc[i]["Text"].find(ans)
                answers.append({
                    "answer_start": [index],
                    "text": [ans]
                })
                contexts.append(mask(df.iloc[i]["Text"]))

            if len(df.iloc[i]["Duration of Stay"]) > 0:
                question = "How long can one stay?"
                questions.append(question)
                ans = df.iloc[i]["Duration of Stay"]
                index = df.iloc[i]["Text"].find(ans)
                answers.append({
                    "answer_start": [index],
                    "text": [ans]
                })
                contexts.append(mask(df.iloc[i]["Text"]))

    return answers, questions, contexts


def mask(text):
    MASK = [
        "Eligibility", "Requirements", "Documents", "Services", "Admission", "Duration",
        "eligibility", "requirements", "documents", "services", "admission", "duration",
        "Eligible", "Requirement", "Document", "Service", "Required",
        "eligible", "requirement", "document", "service", "required",
    ]

    for m in MASK:
        text = text.replace(m, "[MASK]")

    return text


def get_annotated_empty(dfs):
    answers = []
    questions = []
    contexts = []
    for df in dfs:
        for i in range(len(df) - 1):
            if len(df.iloc[i]["Eligibility & Requirements"]) == 0:
                question = "What are the Eligibility & Requirement criteria?"
                questions.append(question)
                ans = ""
                index = 0
                answers.append({
                    "answer_start": [index],
                    "text": [ans]
                })
                contexts.append(mask(df.iloc[i]["Text"]))

            if len(df.iloc[i]["Documents"]) == 0:
                question = "What are the required documents?"
                questions.append(question)
                ans = ""
                index = 0
                answers.append({
                    "answer_start": [index],
                    "text": [ans]
                })
                contexts.append(mask(df.iloc[i]["Text"]))

            if len(df.iloc[i]["Services"]) == 0:
                question = "What are the provided services?"
                questions.append(question)
                ans = ""
                index = 0
                answers.append({
                    "answer_start": [index],
                    "text": [ans]
                })
                contexts.append(mask(df.iloc[i]["Text"]))

            if len(df.iloc[i]["Admission Process"]) == 0:
                question = "What is the admission process?"
                questions.append(question)
                ans = ""
                index = 0
                answers.append({
                    "answer_start": [index],
                    "text": [ans]
                })
                contexts.append(mask(df.iloc[i]["Text"]))

            if len(df.iloc[i]["Duration of Stay"]) == 0:
                question = "How long can one stay?"
                questions.append(question)
                ans = ""
                index = 0
                answers.append({
                    "answer_start": [index],
                    "text": [ans]
                })
                contexts.append(mask(df.iloc[i]["Text"]))

    return answers, questions, contexts
