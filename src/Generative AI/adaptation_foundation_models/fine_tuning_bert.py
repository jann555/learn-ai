'''Exercise: Full-fine tuning BERT In this exercise, you will create a BERT sentiment classifier (actually
DistilBERT) using the Hugging Face Transformers library. You will use the IMDB movie review dataset to complete a
full fine-tuning and evaluate your model.

The IMDB dataset contains movie reviews that are labeled as either positive or negative.'''
# Load the sms_spam dataset
# See: https://huggingface.co/datasets/sms_spam

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
import numpy as np
import pandas as pd
import torch

# The sms_spam dataset only has a train split, so we use the train_test_split method to split it into train and test
dataset = load_dataset("sms_spam", split="train").train_test_split(
    test_size=0.2, shuffle=True, seed=23
)
print("Loaded dataset")

splits = ["train", "test"]

# View the dataset characteristics
print(dataset["train"][0])
'''Pre-process datasets
Now we are going to process our datasets by converting all the text into tokens for our models.'''
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Let's use a lambda function to tokenize all the examples
tokenized_dataset = {}
for split in splits:
    tokenized_dataset[split] = dataset[split].map(
        lambda x: tokenizer(x["sms"], truncation=True), batched=True
    )

# Inspect the available columns in the dataset
print("tokenized_dataset", tokenized_dataset["train"])
# Load and set up the model
# In this case we are doing a full fine tuning, so we will want to unfreeze all parameters.

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    id2label={0: "not spam", 1: "spam"},
    label2id={"not spam": 0, "spam": 1},
)

# Use cuda instead of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)
print(f"Set Device to: {device}")

# Unfreeze all the model parameters.
# Hint: Check the documentation at https://huggingface.co/transformers/v4.2.2/training.html
for param in model.parameters():
    param.requires_grad = True


# print(model)
#Let's train it!
#Now it's time to train our model. We'll use the Trainer class.
#First we'll define a function to compute our accuracy metreic then we make the Trainer.
#In this instance, we will fill in some of the training arguments
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}


# The HuggingFace Trainer class handles the training and eval loop for PyTorch for us.
# Read more about it here https://huggingface.co/docs/transformers/main_classes/trainer
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./data/spam_not_spam",
        # Set the learning rate
        learning_rate=2e-5,
        # Set the per device train batch size and eval batch size
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        # Evaluate and save the model after each epoch
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=1,
        weight_decay=0.01,
        load_best_model_at_end=True,
    ),
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)
print("Initializing trainer")
trainer.train()
# Show the performance of the model on the test set
# What do you think the evaluation accuracy will be?
print("Evaluating trainer")
trainer.evaluate()

#View the results:
# Make a dataframe with the predictions and the text and the labels

items_for_manual_review = tokenized_dataset["test"].select(
    [0, 1, 22, 31, 43, 292, 448, 487]
)

results = trainer.predict(items_for_manual_review)
df = pd.DataFrame(
    {
        "sms": [item["sms"] for item in items_for_manual_review],
        "predictions": results.predictions.argmax(axis=1),
        "labels": results.label_ids,
    }
)
# Show all the cell
pd.set_option("display.max_colwidth", None)
print(df)
