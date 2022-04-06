import pickle
from datasets import Dataset
from utilities import get_annotated_data, get_annotated_empty
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline, Trainer, TrainingArguments
from transformers import DefaultDataCollator
import json
import pandas as pd
import sys
import numpy as np

# model_name = "deepset/roberta-base-squad2"
model_name = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def preprocess_function(data):

    inputs = tokenizer(
        data["questions"],
        data["contexts"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = data["answers"][i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
            #print(idx, len(sequence_ids))
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


# dataset
# with open("../data/qa.json", "r") as file:
#     QA = json.loads(file.read())
#
# answers = []
# questions = []
# contexts = []
# for data in QA:
#     contexts.append(data["context"])
#     questions.append(data["question"])
#     index = data["context"].find(data["answers"])
#     answers.append({
#         "answer_start": [index],
#         "text": [data["answers"]]
#     })

df_1 = pd.read_excel("../data/annotated/annotator_1.xlsx", keep_default_na=False)
df_2 = pd.read_excel("../data/annotated/annotator_2.xlsx", keep_default_na=False)
df_3 = pd.read_excel("../data/annotated/annotator_3.xlsx", keep_default_na=False)
a, q, c = get_annotated_data([df_1[100:], df_2[200:], df_3[200:]])
# questions = np.concatenate((questions, q))
# contexts = np.concatenate((contexts, c))
# answers = np.concatenate((answers, a))
a_null, q_null, c_null = get_annotated_empty([df_1[100:]])
questions = np.concatenate((q_null, q))
contexts = np.concatenate((c_null, c))
answers = np.concatenate((a_null, a))
data = {
    "answers": answers,
    "questions": questions,
    "contexts": contexts
}

my_squad = Dataset.from_dict(data)
tokenized_training = my_squad.map(preprocess_function, batched=True)
# tokenized_training = preprocess_function(questions, contexts, answers)

# validation
df_gold = pd.read_excel("../data/annotated/gold.xlsx", keep_default_na=False)
answers, questions, contexts = get_annotated_data([df_gold])

# tokenized_eval = preprocess_function(questions, contexts, answers)
data = {
    "answers": answers,
    "questions": questions,
    "contexts": contexts
}
my_squad_eval = Dataset.from_dict(data)
tokenized_eval = my_squad_eval.map(preprocess_function, batched=True)

data_collator = DefaultDataCollator()

# learning rate
lr = float(sys.argv[1])

training_args = TrainingArguments(
    output_dir="output",
    evaluation_strategy="epoch",
    learning_rate=lr,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    num_train_epochs=6,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_training,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
with open(str(lr)+"_fine_roberta_6_epoch.pkl", "wb") as file:
    pickle.dump(model, file)
