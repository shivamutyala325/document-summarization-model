from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
import os

model_name='facebook/bart-large-cnn'
folder_path= 'text_summarizer_model'

model=AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer=AutoTokenizer.from_pretrained(model_name)

os.makedirs(folder_path, exist_ok=True)

tokenizer.save_pretrained(folder_path)
model.save_pretrained(folder_path)
print(f'{model_name} downloaded sucessfully to {folder_path}')