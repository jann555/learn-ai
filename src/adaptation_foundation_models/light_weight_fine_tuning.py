# Loading and Evaluating a Foundation Model
# TODO: In the lines below, load your chosen pre-trained Hugging Face model and evaluate its performance
#  prior to fine-tuning. This step includes loading an appropriate tokenizer and dataset.
import numpy as np
from torch import float16
from datasets import load_dataset
from transformers import (AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding,
                          AutoModelForSequenceClassification)

from peft import LoraConfig, get_peft_model, TaskType, AutoPeftModelForSequenceClassification

dataset = load_dataset('google-research-datasets/poem_sentiment', split="train").train_test_split(
    test_size=0.2, shuffle=True, seed=15
)
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

print(dataset["train"][0])


def tokenize_inputs(inputs):
    prompt = inputs["verse_text"]
    tokenizer.truncation_side = "left"
    tokenized_items = tokenizer(
        prompt,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return tokenized_items


tokenized_dataset = {}
splits = ["train", "test"]

for split in splits:
    tokenized_dataset[split] = dataset[split].map(
        lambda x: tokenize_inputs(x), batched=True
    )

print("tokenized_dataset", tokenized_dataset["train"])
print("tokenized_dataset", tokenized_dataset["test"])

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4,
    id2label={0: "negative", 1: "positive", 2: "no impact", 3: "mixed"},
    label2id={"negative": 0, "positive": 1, "no impact": 2, "mixed": 3},
)
model.config.pad_token_id = model.config.eos_token_id
print(model)

# Performing Parameter-Efficient Fine-Tuning
# TODO: In the lines below, create a PEFT model from your loaded model, run a training loop, and save the
#  PEFT model weights.


config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS,
    target_modules=["c_proj"]
)
lora_model = get_peft_model(model, config)
lora_model.print_trainable_parameters()
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=lora_model,
    args=TrainingArguments(
        output_dir="./gpt-lora_results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=6,
        gradient_accumulation_steps=16,
        warmup_steps=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        fp16=True,
        optim="paged_adamw_8bit"
    ),
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)
trainer.train()
print("Evaluating trainer")
trainer.evaluate()

lora_model.save_pretrained("gpt-lora")

# Inference with PEFT
# Loading a Saved PEFT Model
lora_model = AutoPeftModelForSequenceClassification.from_pretrained("gpt-lora", num_labels=4)
tokenizer = AutoTokenizer.from_pretrained(model_name)
ins = tokenizer("and that is why, the lonesome day,", return_tensors="pt")

