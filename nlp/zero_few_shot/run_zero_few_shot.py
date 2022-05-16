from datasets import Dataset
from utilities import get_annotated_data, pack_few_shot, qa_bert_type, annotate
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline, Trainer, TrainingArguments
from transformers import DefaultDataCollator
import pandas as pd
import numpy as np
import sys

# model_name = "deepset/roberta-base-squad2"
# model_name = "ahotrod/electra_large_discriminator_squad2_512"
model_name = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model_file_name = "electra"
# model_file_name = "roberta"
model_file_name = sys.argv[2]
print("MODEL", model_name)
lr = 3e-5
weight_decay = 0.01
epochs = 3


def preprocess_function(data):

    inputs = tokenizer(
        data["questions"],
        data["contexts"],
        # max_length=384,
        # max_length=4096,
        max_length=512,
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


# validation
df_gold = pd.read_excel("../../data/annotated/gold.xlsx", keep_default_na=False)
answers, questions, contexts = annotate([df_gold[:50]], False)

validation = {
    "answers": answers,
    "questions": questions,
    "contexts": contexts
}
dataset = Dataset.from_dict(validation)
tokenized_eval = dataset.map(preprocess_function, batched=True)

# training
df_1 = pd.read_excel("../../data/annotated/annotator_1.xlsx", keep_default_na=False)
df_2 = pd.read_excel("../../data/annotated/annotator_2.xlsx", keep_default_na=False)
df_3 = pd.read_excel("../../data/annotated/annotator_3.xlsx", keep_default_na=False)
eligibility, documents, services, admission, duration = get_annotated_data([df_1[100:], df_2[200:], df_3[200:]])

# few shot
counts = np.arange(0, 110, 10)
for i in counts:
    # load afresh
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    answers, questions, contexts = pack_few_shot(eligibility, documents, services, admission, duration, i)
    print("Loading ", len(answers), "Entries")
    if len(answers) == 0:
        # zero shot
        nlp = pipeline('question-answering', model=model, tokenizer=model_name)
        qa_bert_type(nlp, model_file_name+"0")
    else:
        # train
        train = {
            "answers": answers,
            "questions": questions,
            "contexts": contexts
        }

        dataset = Dataset.from_dict(train)
        tokenized_training = dataset.map(preprocess_function, batched=True)
        data_collator = DefaultDataCollator()

        training_args = TrainingArguments(
            output_dir="output",
            evaluation_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=8,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
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
        nlp = pipeline('question-answering', model=model, tokenizer=model_name)
        qa_bert_type(nlp, model_file_name+str(i))

