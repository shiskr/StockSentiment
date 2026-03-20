# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import os
from azure.identity import DefaultAzureCredential
from azure.mgmt.appcontainers import ContainerAppsAPIClient
from inference.predict import load_model, predict
from scripts.model_registry import promote_staging_model, load_registry
from scripts.metrics import download_metrics
from scripts.model_registry import compare_models
from scripts.metrics import get_training_status

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

@app.post("/admin/train-models")
def trigger_training():
    try:
        subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        if not subscription_id:
            return {"error": "AZURE_SUBSCRIPTION_ID not set"}
        resource_group = "stock-sentiment-rg"
        job_name = "finbert-train-job"

        credential = DefaultAzureCredential()

        client = ContainerAppsAPIClient(
            credential=credential,
            subscription_id=subscription_id
        )

        client.jobs.begin_start(
            resource_group_name=resource_group,
            job_name=job_name
        )

    except Exception as e:
        return {"error": str(e)}

    return {"status": "training started"}

@app.post("/admin/promote-models")
def promote_model():
    version = promote_staging_model()
    return {
        "message": "Model promoted",
        "version": version
    }

@app.get("/admin/latest-training-metrics")
def get_training_metrics():
    registry = load_registry()
    version = registry.get("staging")
    if not version:
        return {"error": "No staging models found"}
    metrics = download_metrics(version)
    return metrics

@app.get("/admin/compare-models")
def compare():
    return compare_models()

@app.post("/admin/auto-promote")
def auto_promote():
    result = compare_models()
    if result.get("decision") == "promote":
        version = promote_staging_model()
        return {
            "status": "promoted",
            "version": version,
            "details": result
        }
    return {
        "status": "not promoted",
        "details": result
    }

@app.get("/admin/training-status")
def training_status():
    return get_training_status()