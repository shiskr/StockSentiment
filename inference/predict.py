from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import json
from azure.storage.blob import BlobServiceClient


_tokenizer = None
_model = None

AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "models"
REGISTRY_PATH = "finbert/registry.json"


def get_production_model_version():
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container = blob_service_client.get_container_client(CONTAINER_NAME)

    blob = container.get_blob_client(REGISTRY_PATH)
    data = blob.download_blob().readall()

    registry = json.loads(data)
    version = registry.get("production")

    if not version:
        version = registry.get("staging")
    return version


def download_model(version):
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container = blob_service_client.get_container_client(CONTAINER_NAME)

    local_dir = f"models/finbert-{version}"
    os.makedirs(local_dir, exist_ok=True)

    blobs = container.list_blobs(name_starts_with=f"finbert/{version}/")

    for blob in blobs:
        filename = blob.name.split("/")[-1]
        local_path = os.path.join(local_dir, filename)

        if filename == "":
            continue

        blob_client = container.get_blob_client(blob.name)
        with open(local_path, "wb") as f:
            f.write(blob_client.download_blob().readall())

    return local_dir

def load_model():
    global _tokenizer, _model

    if _model is None:
        version = get_production_model_version()
        local_model_path = f"models/finbert-{version}"

        if not os.path.exists(local_model_path):
            local_model_path = download_model(version)

        _tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        _model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
        _model.eval()

def predict(text):
    load_model()

    inputs = _tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = _model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    id2label = {0: "SELL", 1: "HOLD", 2: "BUY"}
    return id2label[probs.argmax().item()], probs.tolist()

# preload model on service startup instead of first request
try:
    load_model()
    print("Model loaded successfully at startup.")
except Exception as e:
    print(f"Model failed to load at startup: {e}")