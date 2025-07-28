from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = "all-MiniLM-L6-v2"
SAVE_PATH = "./embedding_model"

os.makedirs(SAVE_PATH, exist_ok=True)

print(f"Downloading model '{MODEL_NAME}'...")

model = SentenceTransformer(MODEL_NAME)

model.save(SAVE_PATH)

print(f"Model successfully downloaded and saved to '{SAVE_PATH}'")