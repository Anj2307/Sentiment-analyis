from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import mlflow.pyfunc
import pandas as pd
import time
import os

app = FastAPI(
    title="Sentiment Analysis API",
    description="Analyzes sentiment of movie/YouTube reviews",
    version="1.0.0"
)

print("Loading model...")

mlflow.set_tracking_uri("sqlite:///mlflow.db")

model =mlflow.pyfunc.load_model(
    model_uri="models:/sentiment-analysis/Production/1"
    )

print("Model loaded successfully!")

class PredictRequest(BaseModel):
    reviews: List[str]

class PredictResponse(BaseModel):
    predictions: List[str]
    positive_percentage: float
    negative_percentage: float
    overall_sentiment: str
    total: int
    inference_time: float

@app.get("/health")
def health():
    return {"status": "ok", "model": "sentiment-analysis", "version": "1.0.0"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):

    if len(request.reviews) == 0:
        raise HTTPException(status_code=400, detail="No reviews provided")
    if len(request.texts)>500:
        raise HTTPException(status_code=400, detail="Too many reviews provided. Maximum is 500.")
    
    start= time.time()
    input_df= pd.DataFrame({"text": request.reviews})
    predictions = model.predict(input_df)
    inference_ms = round((time.time() - start) * 1000, 2)

    total=len(predictions)
    pos_count=predictions.count("positive")
    neg_count=predictions.count("negative")

    positive_pct=round((pos_count/total)*100, 2)
    negative_pct=round((neg_count/total)*100, 2)
    overall_sentiment="positive" if positive_pct > negative_pct else "negative"
    return PredictResponse(
        predictions=predictions,
        positive_percentage=positive_pct,
        negative_percentage=negative_pct,
        overall_sentiment=overall_sentiment,
        total=total,
        inference_time=inference_ms
    )


@app.get("/models/info")
def model_info():
    return {
        "model_name":    "sentiment_model_production",
        "model_version": "1",
        "classes":       ["pos", "neg"],
        "max_texts":     500,
    }