def get_annotated_data(dfs, mask_key_words=False):
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
                if mask_key_words:
                    contexts.append(mask(df.iloc[i]["Text"]))
                else:
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
                if mask_key_words:
                    contexts.append(mask(df.iloc[i]["Text"]))
                else:
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
                if mask_key_words:
                    contexts.append(mask(df.iloc[i]["Text"]))
                else:
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
                if mask_key_words:
                    contexts.append(mask(df.iloc[i]["Text"]))
                else:
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
                if mask_key_words:
                    contexts.append(mask(df.iloc[i]["Text"]))
                else:
                    contexts.append(df.iloc[i]["Text"])

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

    for i in range(len(dfs[0]) - 1):
        if len(dfs[0].iloc[i]["Eligibility & Requirements"].strip()) == \
                len(dfs[1].iloc[i]["Eligibility & Requirements"].strip()) == \
                len(dfs[2].iloc[i]["Eligibility & Requirements"].strip()) == 0:
            question = "What are the Eligibility & Requirement criteria?"
            questions.append(question)
            ans = ""
            index = 0
            answers.append({
                "answer_start": [index],
                "text": [ans]
            })
            contexts.append(dfs[0].iloc[i]["Text"])

        if len(dfs[0].iloc[i]["Documents"].strip()) == len(dfs[1].iloc[i]["Documents"].strip()) == \
                len(dfs[2].iloc[i]["Documents"].strip()) == 0:
            question = "What are the required documents?"
            questions.append(question)
            ans = ""
            index = 0
            answers.append({
                "answer_start": [index],
                "text": [ans]
            })
            contexts.append(dfs[0].iloc[i]["Text"])

        if len(dfs[0].iloc[i]["Services"].strip()) == len(dfs[1].iloc[i]["Services"].strip()) == \
                len(dfs[2].iloc[i]["Services"].strip()) == 0:
            question = "What are the provided services?"
            questions.append(question)
            ans = ""
            index = 0
            answers.append({
                "answer_start": [index],
                "text": [ans]
            })
            contexts.append(dfs[0].iloc[i]["Text"])

        if len(dfs[0].iloc[i]["Admission Process"].strip()) == len(dfs[1].iloc[i]["Admission Process"].strip()) == \
                len(dfs[2].iloc[i]["Admission Process"].strip()) == 0:
            question = "What is the admission process?"
            questions.append(question)
            ans = ""
            index = 0
            answers.append({
                "answer_start": [index],
                "text": [ans]
            })
            contexts.append(dfs[0].iloc[i]["Text"])

        if len(dfs[0].iloc[i]["Duration of Stay"].strip()) == len(dfs[1].iloc[i]["Duration of Stay"].strip()) == \
                len(dfs[2].iloc[i]["Duration of Stay"].strip()) == 0:
            question = "How long can one stay?"
            questions.append(question)
            ans = ""
            index = 0
            answers.append({
                "answer_start": [index],
                "text": [ans]
            })
            contexts.append(dfs[0].iloc[i]["Text"])

    return answers, questions, contexts
