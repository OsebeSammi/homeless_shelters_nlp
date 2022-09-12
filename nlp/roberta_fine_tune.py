import pickle
from datasets import Dataset
from utilities import get_annotated_data, oversample_annotated_data
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments
from transformers import DefaultDataCollator
import pandas as pd
from roberta import RobertaForQuestionAnswering

model_name = "roberta-base"
# model_name = "deepset/roberta-base-squad2"
# model = AutoModelForQuestionAnswering.from_pretrained(model_name)
model = RobertaForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# params
max_length = 512
lr = 1.5e-5
epochs = 2
batch = 4


def preprocess_function(data):

    inputs = tokenizer(
        data["questions"],
        data["contexts"],
        max_length=max_length,
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

        if start_char == 0 and end_char == 0:
            # Answer does not exist
            start_positions.append(0)
            end_positions.append(0)
        # If the answer is not fully inside the context, label it (0, 0)
        elif offset[context_start][0] > end_char or offset[context_end][1] < start_char:
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
answers, questions, contexts = oversample_annotated_data([df_1[100:], df_2[200:], df_3[200:]], False)
# a_none, q_none, c_none = get_annotated_empty([df_1[100:200], df_2[100:200], df_3[100:200]])
# answers = np.concatenate((answers, a_none))
# questions = np.concatenate((questions, q_none))
# contexts = np.concatenate((contexts, c_none))
data = {
    "answers": answers,
    "questions": questions,
    "contexts": contexts
}

my_squad = Dataset.from_dict(data)
tokenized_training = my_squad.map(preprocess_function, batched=True)

# validation
df_gold = pd.read_excel("../data/annotated/gold.xlsx", keep_default_na=False)
answers, questions, contexts = get_annotated_data([df_gold[50:]], False)

data = {
    "answers": answers,
    "questions": questions,
    "contexts": contexts
}
my_squad_eval = Dataset.from_dict(data)
tokenized_eval = my_squad_eval.map(preprocess_function, batched=True)

data_collator = DefaultDataCollator()

training_args = TrainingArguments(
    output_dir="output",
    evaluation_strategy="epoch",
    learning_rate=lr,
    per_device_train_batch_size=batch,
    per_device_eval_batch_size=batch,
    num_train_epochs=epochs,
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
with open("models/roberta_base_cross", "wb") as file:
    pickle.dump(model, file)
