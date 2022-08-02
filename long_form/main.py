import pickle
import json
from datasets import load_dataset
from transformers import AutoTokenizer, DefaultDataCollator, Trainer, TrainingArguments, pipeline
from qa_head import RobertaForQuestionAnswering
import nlp.evaluate as squad_eval
from tqdm import tqdm
import torch
import sys

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
LOCAL = bool(sys.argv[2])


def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    context = examples["context"]
    answers = examples["answers"]

    if LOCAL:
        # use 10% of the data for local
        length = int(0.005 * len(questions))
        questions = questions[:length]
        context = context[:length]
        answers = answers[:length]

    inputs = tokenizer(
        questions,
        context,
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]

        # no answer
        if len(answer["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
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


model = RobertaForQuestionAnswering.from_pretrained(model_name)
squad = load_dataset("squad_v2")
tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)
data_collator = DefaultDataCollator()

# device
# if torch.has_mps:
#     device = torch.device('mps')
# else:
#     device = torch.device('cpu')
# model.to(device)
# print("MOVING TO", device)

# train
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# save
with open(str(sys.argv[1]), "wb") as file:
    pickle.dump(model, file)

# test
# device = torch.device('cpu')
# print("MOVING TO", device)
# model.to(device)

nlp = pipeline('question-answering', model=model, tokenizer="roberta-base")

with open(str(sys.argv[3]), "r") as file:
    squad_dev = json.loads(file.read())

pred = {}
squad_dev["data"] = squad_dev["data"][:10]
squad_dev = squad_dev["data"]
for i in tqdm(range(len(squad_dev))):
    for j in range(len(squad_dev[i]["paragraphs"])):
        for k in range(len(squad_dev[i]["paragraphs"][j]["qas"])):
            qa = {
                "question": squad_dev[i]["paragraphs"][j]["qas"][k]["question"],
                "context": squad_dev[i]["paragraphs"][j]["context"]
            }

            answer = nlp(qa)

            if answer["score"] < 0.1:
                # no answer
                pred[squad_dev[i]["paragraphs"][j]["qas"][k]["id"]] = ""
            else:
                # has answer
                pred[squad_dev[i]["paragraphs"][j]["qas"][k]["id"]] = answer["answer"]


# results = squad_eval.start("squad_dev.json", "squad_pred.json")
results = squad_eval.run(squad_dev, pred)
print(results)
