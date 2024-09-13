from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# Load the pre-trained model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the LoRa configuration
lora_config = LoraConfig(
    r=8,  # Rank of the low-rank adaptation
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,  # Dropout rate
    target_modules=["classifier.dense.weight", "classifier.dense.bias"]  # Target modules to apply LoRa
)

# Apply the LoRa adapter to the model
model = get_peft_model(model, lora_config)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Define a simple dataset for demonstration purposes
train_texts = ["I love this!", "I hate this!"]
train_labels = [1, 0]
val_texts = ["This is great!", "This is terrible!"]
val_labels = [1, 0]

# Tokenize the dataset
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# Create a Dataset object
import torch
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = SimpleDataset(train_encodings, train_labels)
val_dataset = SimpleDataset(val_encodings, val_labels)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()
