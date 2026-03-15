# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from inference.predict import load_model, predict
from scripts.model_registry import promote_staging_model

app = FastAPI(title="Stock Sentiment API")

class TextRequest(BaseModel):
    text: str

@app.on_event("startup")
def startup_event():
    load_model()

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def get_prediction(request: TextRequest):
    result = predict(request.text)
    return result

@app.post("/admin/promote-model")
def promote_model():
    version = promote_staging_model()
    return {
        "message": "Model promoted",
        "version": version
    }