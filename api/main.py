from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import mlflow
import mlflow.pyfunc
import pandas as pd
import time
import os
from src.collect import fetch_comments_from_url
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

app = FastAPI(
    title="Sentiment Analysis API",
    description="Analyzes sentiment of movie/YouTube reviews",
    version="1.0.0"
)

# load model once at startup
print("Loading model...")
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.pyfunc.load_model(
    model_uri="models:/distilbert-imdb-sentiment/3"
)
print("Model loaded successfully!")

# ── schemas ───────────────────────────────────────────────

class PredictRequest(BaseModel):
    reviews: List[str]

class PredictResponse(BaseModel):
    predictions:         List[str]
    positive_percentage: float
    negative_percentage: float
    overall_sentiment:   str
    total:               int
    inference_time:      float

class YouTubeRequest(BaseModel):
    url:          str
    max_comments: int = 100

class YouTubeResponse(BaseModel):
    video_id:     str
    total:        int
    positive_pct: float
    negative_pct: float
    overall:      str
    top_positive: str
    top_negative: str
    inference_ms: float
    comments:     List[dict]

class DirectBertModel:
    def __init__(self):
        local_path="results/bert_sentiment_model"
        hub_path="Anj2307/youtube-sentiment-distilbert"

        if os.path.exists(local_path):
            print(f"Loading model from local path: {local_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(local_path)
            self.model     = AutoModelForSequenceClassification.from_pretrained(local_path)
        else:
            print(f"Local path not found. Loading model from Hugging Face Hub: {hub_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(hub_path)
            self.model     = AutoModelForSequenceClassification.from_pretrained(hub_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    def predict(self, input_df):
        texts=input_df["text"].tolist()
        results=[]

        for text in texts:
            inputs= self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt"

            )
            with torch.no_grad():
                outputs=self.model(**inputs)
                pred= torch.argmax(outputs.logits,dim=1).item()
                results.append("pos" if pred==0 else "neg")
        return results

model=DirectBertModel()



# ── endpoints ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":  "ok",
        "model":   "sentiment-analysis",
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):

    if len(request.reviews) == 0:
        raise HTTPException(
            status_code=400,
            detail="No reviews provided"
        )
    if len(request.reviews) > 500:
        raise HTTPException(
            status_code=400,
            detail="Too many reviews. Maximum is 500."
        )

    start        = time.time()
    input_df     = pd.DataFrame({"text": request.reviews})
    predictions  = list(model.predict(input_df))
    inference_ms = round((time.time() - start) * 1000, 2)

    total        = len(predictions)
    pos_count    = predictions.count("pos")
    neg_count    = predictions.count("neg")
    positive_pct = round((pos_count / total) * 100, 2)
    negative_pct = round((neg_count / total) * 100, 2)
    overall      = "positive" if pos_count >= neg_count else "negative"

    return PredictResponse(
        predictions=predictions,
        positive_percentage=positive_pct,
        negative_percentage=negative_pct,
        overall_sentiment=overall,
        total=total,
        inference_time=inference_ms
    )

@app.post("/analyze/youtube", response_model=YouTubeResponse)
def analyze_youtube(request: YouTubeRequest):

    # step 1: extract video id
    if "v=" in request.url:
        video_id = request.url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in request.url:
        video_id = request.url.split("youtu.be/")[1].split("?")[0]
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid YouTube URL"
        )

    # step 2: fetch comments
    try:
        raw_comments = fetch_comments_from_url(
            request.url,
            max_comments=request.max_comments
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if len(raw_comments) == 0:
        raise HTTPException(
            status_code=404,
            detail="No comments found. Video may be private or comments disabled."
        )

    # step 3: run model
    start        = time.time()
    texts        = [c["text"] for c in raw_comments]
    input_df     = pd.DataFrame({"text": texts})
    predictions  = list(model.predict(input_df))
    inference_ms = round((time.time() - start) * 1000, 2)

    # step 4: aggregate
    total        = len(predictions)
    pos_count    = predictions.count("pos")
    neg_count    = predictions.count("neg")
    positive_pct = round(pos_count / total * 100, 1)
    negative_pct = round(neg_count / total * 100, 1)
    overall      = "positive" if pos_count >= neg_count else "negative"

    # step 5: combine comments with predictions
    comments_with_pred = []
    for comment, pred in zip(raw_comments, predictions):
        comments_with_pred.append({
            "text":      comment["text"],
            "likes":     comment["likes"],
            "date":      comment["date"],
            "sentiment": pred
        })

    # step 6: find top comments by likes
    pos_comments = [c for c in comments_with_pred if c["sentiment"] == "pos"]
    neg_comments = [c for c in comments_with_pred if c["sentiment"] == "neg"]

    top_positive = max(pos_comments, key=lambda x: x["likes"])["text"] \
                   if pos_comments else "No positive comments found"
    top_negative = max(neg_comments, key=lambda x: x["likes"])["text"] \
                   if neg_comments else "No negative comments found"

    return YouTubeResponse(
        video_id     = video_id,
        total        = total,
        positive_pct = positive_pct,
        negative_pct = negative_pct,
        overall      = overall,
        top_positive = top_positive,
        top_negative = top_negative,
        inference_ms = inference_ms,
        comments     = comments_with_pred
    )

@app.get("/models/info")
def model_info():
    return {
        "model_name":    "distilbert-imdb-sentiment",
        "model_version": "3",
        "classes":       ["pos", "neg"],
        "max_texts":     500,
    }