from transformers import AutoTokenizer,AutoModelForQuestionAnswering
import os

model_name="deepset/roberta-base-squad2"
folder_path= 'QandA_model'

model=AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer=AutoTokenizer.from_pretrained(model_name)

os.makedirs(folder_path, exist_ok=True)

tokenizer.save_pretrained(folder_path)
model.save_pretrained(folder_path)
print(f'{model_name} downloaded sucessfully to {folder_path}')