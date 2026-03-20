from azure.storage.blob import BlobServiceClient
from datetime import datetime
import json
import os

def download_metrics(version: str):
    """Download metrics.json for a given models version from Azure Blob"""
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container = blob_service.get_container_client("models")

    blob_path = f"finbert/{version}/metrics.json"
    blob = container.get_blob_client(blob_path)

    try:
        data = blob.download_blob().readall()
        return json.loads(data)
    except Exception:
        return {"error": f"metrics not found for version {version}"}

def upload_status(status_data: dict):
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        raise Exception("AZURE_STORAGE_CONNECTION_STRING not set")
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_name = "models"  # ⚠️ change if needed
    blob_path = "finbert/training_status/latest.json"
    container_client = blob_service_client.get_container_client(container_name)
    # enrich status
    status_data["timestamp"] = datetime.utcnow().isoformat()
    blob_client = container_client.get_blob_client(blob_path)
    blob_client.upload_blob(
        json.dumps(status_data),
        overwrite=True
    )
    print("Training status updated:", status_data)

def get_training_status():
    blob_service_client = BlobServiceClient.from_connection_string(
        os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    )
    container = blob_service_client.get_container_client("models")
    blob_path = "finbert/training_status/latest.json"
    try:
        blob_client = container.get_blob_client(blob_path)
        data = blob_client.download_blob().readall()
        if not data:
            return {"status": "unknown", "message": "No training status yet"}
        return json.loads(data)
    except Exception as e:
        return {"status": "error", "error": str(e)}