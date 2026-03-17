import json
import os
from azure.storage.blob import BlobServiceClient

def promote_staging_model():
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container = blob_service_client.get_container_client("models")
    blob = container.get_blob_client("finbert/registry.json")
    registry = json.loads(blob.download_blob().readall().decode("utf-8"))
    if not registry["staging"]:
        raise Exception("No staging model available")
    registry["production"] = registry["staging"]
    blob.upload_blob(json.dumps(registry), overwrite=True)
    return registry["production"]


def load_registry():
    """Load model registry from Azure Blob Storage"""
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container = blob_service_client.get_container_client("models")
    blob = container.get_blob_client("finbert/registry.json")
    registry = json.loads(blob.download_blob().readall().decode("utf-8"))
    return registry