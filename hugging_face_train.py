import argparse

import wandb
from datasets import Dataset
from transformers import (BartForConditionalGeneration, BartTokenizer, EarlyStoppingCallback,
                          T5ForConditionalGeneration,
                          T5Tokenizer,
                          Trainer, TrainingArguments)

from helpers import get_data_from_file

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model name")
args = parser.parse_args()

"""
models:
prithivida/grammar_error_correcter_v1
grammarly/coedit-large

pszemraj/bart-base-grammar-synthesis
oliverguhr/spelling-correction-english-base

vennify/t5-base-grammar-correction
"""

models = {"T5":
              ["prithivida/grammar_error_correcter_v1",
               "grammarly/coedit-large", ],
          "BART":
              ["pszemraj/bart-base-grammar-synthesis",
               "oliverguhr/spelling-correction-english-base", ],
          "happy":
              ["vennify/t5-base-grammar-correction"], }

name = args.model.replace("/", "-")
wandb.init(project="finetuning-" + name, name=name)

if args.model in models["T5"]:
    model = T5ForConditionalGeneration.from_pretrained(args.model)
    tokenizer = T5Tokenizer.from_pretrained(args.model)
elif args.model in models["BART"]:
    model = BartForConditionalGeneration.from_pretrained(args.model)
    tokenizer = BartTokenizer.from_pretrained(args.model)
elif args.model in models["happy"]:
    from happytransformer import HappyTextToText

    happy_tt = HappyTextToText("T5", args.model)
    model = happy_tt.model
    tokenizer = happy_tt.tokenizer

else:
    raise ValueError("Model not found")

# load and preprocess data
train_data = get_data_from_file("train")
dev_data = get_data_from_file("dev")

train_dataset = Dataset.from_dict({"source": train_data[0], "target": train_data[1]})
dev_dataset = Dataset.from_dict({"source": dev_data[0], "target": dev_data[1]})


def preprocess_function(data):
    inputs = data['source']
    targets = data['target']
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length')

    # remove pad tokens from loss calculation
    labels["input_ids"] = [
            label if label != tokenizer.pad_token_id else -100 for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


train_dataset = train_dataset.map(preprocess_function, batched=True)
dev_dataset = dev_dataset.map(preprocess_function, batched=True)

# train
training_args = TrainingArguments(
        output_dir="./" + name + "-finetuned",
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        logging_dir="./logs",
        logging_steps=10,
)

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()

model.save_pretrained("./" + name + "-finetuned")
tokenizer.save_pretrained("./" + name + "-finetuned")

wandb.finish()
