# Loading and Evaluating a Foundation Model
# TODO: In the lines below, load your chosen pre-trained Hugging Face model and evaluate its performance
#  prior to fine-tuning. This step includes loading an appropriate tokenizer and dataset.
import numpy as np
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments, DataCollatorWithPadding, GPT2Tokenizer, AutoModelForCausalLM)
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM

dataset = load_dataset('google-research-datasets/poem_sentiment', split="train").train_test_split(
    test_size=0.2, shuffle=True, seed=23
)
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenized_dataset = {}
splits = ["train", "test"]
print(dataset["train"][0])
for split in splits:
    tokenized_dataset[split] = dataset[split].map(
        lambda x: tokenizer(x["verse_text"], truncation=True, padding="max_length", max_length=8), batched=True
    )

print("tokenized_dataset", tokenized_dataset["train"])

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    num_labels=4,
    id2label={0: "negative", 1: "positive", 2: "no impact", 3: "mixed"},
    label2id={"negative": 0, "positive": 1, "no impact": 2, "mixed": 3}
)

# Performing Parameter-Efficient Fine-Tuning
# TODO: In the lines below, create a PEFT model from your loaded model, run a training loop, and save the
#  PEFT model weights.


config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules="all-linear",
    bias="none",
    lora_dropout=0.1
)
lora_model = get_peft_model(model, config)
lora_model.print_trainable_parameters()


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    return {"accuracy": (predictions == labels).mean()}


trainer = Trainer(
    model=lora_model,
    args=TrainingArguments(
        output_dir="./lora_results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=1,
        weight_decay=0.01
    ),
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)
trainer.train()
print("Evaluating trainer")
trainer.evaluate()

lora_model.save_pretrained("bert-lora")

# Inference with PEFT
# Loading a Saved PEFT Model
lora_model = AutoModelForCausalLM.from_pretrained("bert-lora", ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer("and that is why, the lonesome day,", return_tensors="pt")
outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=8)
print(tokenizer.batch_decode(outputs))

