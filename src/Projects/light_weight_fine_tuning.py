# Section imports
from datasets import load_dataset
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, \
    DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, TaskType
from peft import AutoPeftModelForSequenceClassification
import torch
import evaluate

# Create variables and constants
FOUNDATION_MODEL = "gpt2"
tokenized_dataset = {}
splits = ["train", "test"]

# Downloading the date set
column_label = "verse_text"
dataset_name = 'poem_sentiment'
dataset = load_dataset(dataset_name, split='train').train_test_split(test_size=0.2, shuffle=True, seed=24)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Defined function to process tokenized inputs

def tokenized_inputs(inputs, tokenizer):
    prompt = inputs[column_label]
    tokenizer.truncation_side = "left"
    tokenized_items = tokenizer(
        prompt,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )
    return tokenized_items


# Defined function to evaluate accuracy of model
# https://huggingface.co/spaces/evaluate-metric/precision
def compute_metrics(eval_preds):
    metric = evaluate.load("precision")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="micro")


# Created Generic function to return the Trainer object
def get_trainer(trainer_model, dataset_name, metrics, num_epoch, output_dir, tokenizer):
    trained_model = Trainer(
        model=trainer_model,
        args=TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=96,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=num_epoch,
            weight_decay=0.01,
            load_best_model_at_end=True
        ),
        train_dataset=dataset_name["train"],
        eval_dataset=dataset_name["test"],
        tokenizer=tokenizer,
        compute_metrics=metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    return trained_model


# Define Tokenizer function
def get_tokenizer(modal_name):
    tokenizer = AutoTokenizer.from_pretrained(modal_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# Configuring the tokenizer and adding padding tokens function
tokenizer_model = get_tokenizer(FOUNDATION_MODEL)

# Tokenize inputs
for split in splits:
    tokenized_dataset[split] = dataset[split].map(
        lambda x: tokenized_inputs(x, tokenizer_model), batched=True
    )

print("tokenized_dataset", tokenized_dataset["train"])

# Create Label objects to pass down to ouw model
id2label = {0: "negative", 1: "positive", 2: "no impact", 3: "mixed"}
label2id = {"negative": 0, "positive": 1, "no impact": 2, "mixed": 3}

# Loading the foundation model with the correct number of labels and identifiers
model = AutoModelForSequenceClassification.from_pretrained(
    FOUNDATION_MODEL,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
)

model.config.pad_token_id = model.config.eos_token_id
model.resize_token_embeddings(len(tokenizer_model))
model.to(device)

# Created Trainer Object and loaded the original model
gpt2_model_trainer = get_trainer(
    trainer_model=model,
    dataset_name=tokenized_dataset,
    metrics=compute_metrics,
    num_epoch=1,
    output_dir="./",
    tokenizer=tokenizer_model
)
print(f'Evaluating foundation model...')
original_model_eval = gpt2_model_trainer.evaluate()
print(original_model_eval)

# Configure LoRa settings
targets = ["c_proj", "c_fc", "c_attn"]
config = LoraConfig(
    r=1,
    lora_alpha=8,
    lora_dropout=0.01,
    bias="none",
    task_type=TaskType.SEQ_CLS,
    target_modules=targets,
    fan_in_fan_out=True
)
# Create Peft Model
lora_model = get_peft_model(model, config)
lora_model.print_trainable_parameters()

# Freeze All parameters except those being fine tunes
for name, param in lora_model.named_parameters():
    if any(target in name for target in targets):
        param.requires_grad = True
    else:
        param.requires_grad = False

# Define Train Object Using Lora Model from Peft
lora_trainer = get_trainer(
    trainer_model=lora_model,
    dataset_name=tokenized_dataset,
    metrics=compute_metrics,
    num_epoch=1,
    output_dir="./gpt-lora",
    tokenizer=tokenizer_model
)
lora_trainer.train()
# Print lora_trainer evaluation
eval_lora = lora_trainer.evaluate()
print(eval_lora)

lora_model.save_pretrained("gpt-lora-trained")

# Loading saved PEFT model from directory
lora_peft_model = AutoPeftModelForSequenceClassification.from_pretrained("gpt-lora-trained", num_labels=4)
lora_peft_model.config.pad_token_id = lora_peft_model.config.eos_token_id
lora_peft_model.resize_token_embeddings(len(tokenizer_model))
lora_peft_model.to(device)
lora_model_trainer = get_trainer(
    trainer_model=lora_peft_model,
    dataset_name=tokenized_dataset,
    metrics=compute_metrics,
    num_epoch=1,
    output_dir="./",
    tokenizer=tokenizer_model
)
lora_model_eval = lora_model_trainer.evaluate()
print(f'Lora Eval {lora_model_eval}')

# Starting Inference
input_text = "in prosperous days. like a dim, waning lamp"
model_inputs = tokenizer_model(input_text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = lora_peft_model(**model_inputs)
    results = torch.argmax(outputs.logits, dim=-1)
# Converting Prediction to Label
predicted_label = id2label[results.item()]

# Input Text with Predicted outout
print(f"Input: {input_text}\n Predicted label: {predicted_label}")

# comparing Evaluation Results
print(f'Original model eval: {original_model_eval} \nLora Model Eval: {lora_model_eval}')
