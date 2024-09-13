# Loading and Evaluating a Foundation Model
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding, AutoModelForSequenceClassification
)

from peft import LoraConfig, get_peft_model, TaskType, AutoPeftModelForSequenceClassification

model_name = "gpt2"
dataset_path = 'poem_sentiment'

# Select cuda instead of cpu when available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def get_tokenizer(model_param):
    tokenizer_object = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer_object.pad_token = tokenizer_object.eos_token
    return tokenizer_object


def tokenize_inputs(inputs):
    prompt = inputs["verse_text"]
    tokenizer.truncation_side = "left"
    tokenized_items = tokenizer(
        prompt,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    )
    return tokenized_items


tokenizer = get_tokenizer(model_name)

tokenized_dataset = {}
splits = ["train", "test"]
dataset = load_dataset(dataset_path, split="train").train_test_split(
    test_size=0.2, shuffle=True, seed=24
)

print(dataset["train"][0])

for split in splits:
    tokenized_dataset[split] = dataset[split].map(
        lambda x: tokenize_inputs(x), batched=True
    )

print("tokenized_dataset", tokenized_dataset["train"])

id2label = {0: "negative", 1: "positive", 2: "no impact", 3: "mixed"}
label2id = {"negative": 0, "positive": 1, "no impact": 2, "mixed": 3}

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)
model.config.pad_token_id = model.config.eos_token_id
model.resize_token_embeddings(len(tokenizer))
print(model)
# Freezing weights for pre-training and fine-tuning
for param in model.parameters():
    param.requires_grad = False
# Performing Parameter-Efficient Fine-Tuning

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    return {"accuracy": (predictions == labels).mean()}


def get_trainer(trainer_model, dataset_name, metrics, num_epoch, output_dir):
    trained_model = Trainer(
        model=trainer_model,
        args=TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=num_epoch,
            weight_decay=0.01,
            load_best_model_at_end=True
        ),
        train_dataset=dataset_name["train"],
        eval_dataset=dataset_name["test"],
        tokenizer=tokenizer,
        compute_metrics=metrics,
        data_collator=data_collator,
    )
    return trained_model


gpt2_model_trainer = get_trainer(
    trainer_model=model,
    dataset_name=tokenized_dataset,
    metrics=None,
    num_epoch=1,
    output_dir="./model"
)
print('Evaluating original model')
original_model_eval = gpt2_model_trainer.evaluate()

config = LoraConfig(
    r=1,
    lora_alpha=16,
    lora_dropout=0.01,
    bias="lora_only",
    task_type=TaskType.SEQ_CLS,
    target_modules=["c_proj", "c_fc", "c_attn"],
    use_rslora=True,
    fan_in_fan_out=True
)
lora_model = get_peft_model(model, config)
lora_model.print_trainable_parameters()

lora_trained = get_trainer(
    trainer_model=lora_model,
    dataset_name=tokenized_dataset,
    metrics=compute_metrics,
    num_epoch=4,
    output_dir="./gpt-lora"
)
print('Training model')
lora_trained.train()
print('Evaluating lora model')
lora_model_eval = lora_trained.evaluate()

lora_model.save_pretrained("gpt-lora")

# Inference with PEFT
# Loading a Saved PEFT Model
lora_model = AutoPeftModelForSequenceClassification.from_pretrained(
    "gpt-lora", num_labels=4)
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_text = "in prosperous days. like a dim, waning lamp"
model_inputs = tokenizer(input_text, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = lora_model(**model_inputs)
    result = torch.argmax(outputs.logits, dim=-1)

# Convert prediction to label
predicted_label = id2label[result.item()]

print(f"Input: {input_text}\n Predicted label: {predicted_label}")
print(f'Original model eval: {original_model_eval} \nLora Model Eval: {lora_model_eval}')
