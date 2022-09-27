import pickle
import json
from datasets import load_dataset
from transformers import AutoTokenizer, DefaultDataCollator, Trainer, TrainingArguments, pipeline
from qa_head import RobertaForQuestionAnswering
import evaluate as squad_eval
from tqdm import tqdm
import torch
import sys

param_path = str(sys.argv[1])
with open(param_path, "r") as file:
    parameters = json.load(file)

DATA = parameters["data"]
model_name = parameters["model"]
name_trained = str(parameters["pool"]) + "_" + str(parameters["mode"])
name_trained = name_trained + "_scale" if parameters["scale"] else name_trained + "_context"
name_trained = name_trained + "_" + str(parameters["no_answer"]) + "_" + str(parameters["lr"]) + "_" + \
               str(parameters["epochs"])

tokenizer = AutoTokenizer.from_pretrained(model_name)
no_answer = "No answer"
space = ". "

with open("token_map.json", "r") as file:
    synonyms = json.loads(file.read())


def synonym_er(text):
    words = text.split(" ")
    synonym_ed = ""
    for word in words:
        if word in synonyms:
            synonym_ed += " " + synonyms[word]
        else:
            synonym_ed += " " + word
    return synonym_ed


def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    context = examples["context"]
    answers = examples["answers"]

    synonym_question = []
    synonym_context = []
    for i in range(len(questions)):
        q = synonym_er(questions[i])
        synonym_question.append(q)

        c = synonym_er(context[i])
        synonym_context.append(c)

    if DATA != 1.0:
        # use 10% of the data for local
        length = int(DATA * len(questions))
        questions = questions[:length]
        context = context[:length]
        answers = answers[:length]
        synonym_question = synonym_question[:length]
        synonym_context = synonym_context[:length]

    if parameters["no_answer"]:
        # change null answers from pointing to CLS
        for i, answer in enumerate(answers):
            context[i] = no_answer + space + context[i]
            if len(answer["answer_start"]) == 0:
                answers[i] = {'text': [no_answer], 'answer_start': [0]}
            else:
                for j, ans_start in enumerate(answer["answer_start"]):
                    answers[i]["answer_start"][j] = len(no_answer) + len(space) + answer["answer_start"][j]

    inputs = tokenizer(
        questions,
        context,
        max_length=parameters["max_length"],
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    synonyms = tokenizer(
        synonym_question,
        synonym_context,
        max_length=parameters["max_length"],
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
            # start_positions.append(0)
            # end_positions.append(0)
            start_char = 0
            end_char = 0
        else:
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

        if start_char == 0 and end_char == 0:
            # Answer does not exist
            start_positions.append(0)
            end_positions.append(0)
        elif offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            # If the answer is not fully inside the context, label it (0, 0)
            if parameters["no_answer"]:
                # using no answer token
                start_positions.append(context_start)
                end_positions.append(context_start + 1)
            else:
                # If the answer is not fully inside the context, label it (0, 0)
                # use [CLS]
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
    inputs["synonyms"] = synonyms.data["input_ids"]
    return inputs


model = RobertaForQuestionAnswering.from_pretrained(model_name)
squad = load_dataset("squad_v2")
tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)
data_collator = DefaultDataCollator()

# move to most 'fastest' device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device('cpu')
    # if torch.has_mps:
    #     device = torch.device('mps')
    # else:
    #     device = torch.device('cpu')
model.to(device)

# train
training_args = TrainingArguments(
    output_dir="./results_"+name_trained,
    evaluation_strategy="epoch",
    learning_rate=parameters["lr"],
    per_device_train_batch_size=parameters["batch"],
    per_device_eval_batch_size=parameters["batch"],
    num_train_epochs=parameters["epochs"],
    weight_decay=0.01,
    save_strategy="steps",
    save_steps=10000
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

nlp = pipeline('question-answering', model=model, tokenizer="roberta-base")

with open(parameters["dev"], "r") as file:
    squad_dev = json.loads(file.read())

pred = {}
# squad_dev["data"] = squad_dev["data"][:10]
squad_dev = squad_dev["data"]
for i in tqdm(range(len(squad_dev))):
    for j in range(len(squad_dev[i]["paragraphs"])):
        for k in range(len(squad_dev[i]["paragraphs"][j]["qas"])):
            if parameters["no_answer"]:
                qa = {
                    "question": squad_dev[i]["paragraphs"][j]["qas"][k]["question"],
                    "context": no_answer + space + squad_dev[i]["paragraphs"][j]["context"]
                }

                answer = nlp(qa)

                if no_answer in answer["answer"]:
                    # no answer
                    pred[squad_dev[i]["paragraphs"][j]["qas"][k]["id"]] = ""
                else:
                    # has answer
                    pred[squad_dev[i]["paragraphs"][j]["qas"][k]["id"]] = answer["answer"]
            else:
                qa = {
                    "question": squad_dev[i]["paragraphs"][j]["qas"][k]["question"],
                    "context": squad_dev[i]["paragraphs"][j]["context"]
                }

                answer = nlp(qa)

                pred[squad_dev[i]["paragraphs"][j]["qas"][k]["id"]] = answer["answer"]


# results = squad_eval.start("squad_dev.json", "squad_pred.json")
results = squad_eval.run(squad_dev, pred)
print(results)

# save
model.to(torch.device('cpu'))
with open(name_trained, "wb") as file:
    pickle.dump(model, file)
print(parameters)
