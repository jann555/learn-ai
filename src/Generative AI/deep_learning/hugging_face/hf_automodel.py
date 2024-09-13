'''🤗 Transformers provides a simple and unified way to load pretrained instances. This means you can load an
AutoModel like you would load an AutoTokenizer. The only difference is selecting the correct AutoModel for the task.
For text (or sequence) classification, you should load AutoModelForSequenceClassification:'''
from transformers import AutoModelForSequenceClassification

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)

pt_outputs = pt_model(**pt_batch)
