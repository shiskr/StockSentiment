import json
import os
from azure.storage.blob import BlobServiceClient

def download_blob(blob_name: str) -> str:
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container = blob_service_client.get_container_client("models")
    blob = container.get_blob_client(blob_name)
    return blob.download_blob().readall().decode("utf-8")


def promote_staging_model():
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container = blob_service_client.get_container_client("models")
    blob = container.get_blob_client("finbert/registry.json")
    registry = json.loads(blob.download_blob().readall().decode("utf-8"))
    if not registry["staging"]:
        raise Exception("No staging models available")
    registry["production"] = registry["staging"]
    blob.upload_blob(json.dumps(registry), overwrite=True)
    return registry["production"]


def load_registry():
    """Load models registry from Azure Blob Storage"""
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container = blob_service_client.get_container_client("models")
    blob = container.get_blob_client("finbert/registry.json")
    registry = json.loads(blob.download_blob().readall().decode("utf-8"))
    return registry

def get_model_metrics(version):
    blob_name = f"finbert/{version}/metrics.json"
    data = download_blob(blob_name)
    return json.loads(data)

def compare_models():
    registry = load_registry()

    prod = registry.get("production")
    staging = registry.get("staging")

    if not staging:
        return {"error": "No staging models"}

    staging_metrics = get_model_metrics(staging)

    if not prod:
        return {
            "decision": "promote",
            "reason": "no production models",
            "staging_metrics": staging_metrics
        }

    prod_metrics = get_model_metrics(prod)

    if staging_metrics.get("accuracy", 0) > prod_metrics.get("accuracy", 0):
        return {
            "decision": "promote",
            "staging": staging_metrics,
            "production": prod_metrics
        }

    return {
        "decision": "reject",
        "staging": staging_metrics,
        "production": prod_metrics
    }