from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

# Creating a PEFT Config
config = LoraConfig()
# Converting a Transformers Model into a PEFT Model
model = AutoModelForCausalLM.from_pretrained("gpt2")
lora_model = get_peft_model(model, config)
# Training with a PEFT Model
lora_model.print_trainable_parameters()
# Saving a Trained PEFT Model
lora_model.save_pretrained("gpt-lora")

# Inference with PEFT
# Loading a Saved PEFT Model
lora_model = AutoModelForCausalLM.from_pretrained("gpt-lora")
# Generating Text from a PEFT Model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
inputs = tokenizer("Hello, my name is ", return_tensors="pt")
outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
print(tokenizer.batch_decode(outputs))
