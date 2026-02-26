# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from inference.predict import load_model, predict

app = FastAPI(title="Stock Sentiment API")

class TextRequest(BaseModel):
    text: str

@app.on_event("startup")
def startup_event():
    load_model()

@app.post("/predict")
def get_prediction(request: TextRequest):
    result = predict(request.text)
    return result

@app.get("/")
def health():
    return {"status": "ok"}