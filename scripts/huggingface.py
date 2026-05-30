from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()
login(token=os.getenv("HUGGINGFACE_API_KEY"))

# correct relative path from scripts/ to results/
model_path = "results/bert_sentiment_model"

# verify path exists before pushing
print(f"Path exists: {os.path.exists(model_path)}")
print(f"Files: {os.listdir(model_path)}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model     = AutoModelForSequenceClassification.from_pretrained(model_path)

tokenizer.push_to_hub("Anj2307/youtube-sentiment-distilbert")
model.push_to_hub("Anj2307/youtube-sentiment-distilbert")

print("Done! Check huggingface.co/Anj2307/youtube-sentiment-distilbert")