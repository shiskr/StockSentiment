from azure.storage.blob import BlobServiceClient
import json
import os

def download_metrics(version: str):
    """Download metrics.json for a given model version from Azure Blob"""
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