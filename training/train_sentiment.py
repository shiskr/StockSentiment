import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from azure.storage.blob import BlobServiceClient
from datetime import datetime, timezone, timedelta

from scripts.metrics import upload_status

IST = timezone(timedelta(hours=5, minutes=30))
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

def load_config(path: str) -> dict:
    """Load YAML training configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load processed sentiment dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    return pd.read_csv(csv_path)

def download_dataset_from_blob(container_name, blob_name, local_path, connection_string):
    """Download dataset from Azure Blob Storage"""
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container = blob_service_client.get_container_client(container_name)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    with open(local_path, "wb") as file:
        blob_client = container.get_blob_client(blob_name)
        download_stream = blob_client.download_blob()
        file.write(download_stream.readall())

    print(f"Dataset downloaded to {local_path}")

def load_dataset_registry(container_name, connection_string):
    """Load dataset registry from Azure Blob"""
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container = blob_service_client.get_container_client(container_name)

    registry_blob = "datasets/registry.json"

    try:
        blob_client = container.get_blob_client(registry_blob)
        data = blob_client.download_blob().readall()
        registry = json.loads(data)
    except Exception:
        raise RuntimeError("Dataset registry not found in blob storage")

    return registry

def get_dataset_blob_path(connection_string):
    """
    Resolve which dataset should be used for training.
    Priority:
    1. DATASET_BLOB env variable
    2. dataset registry 'latest'
    """
    dataset_blob = os.getenv("DATASET_BLOB")

    if dataset_blob:
        print(f"Training dataset: {dataset_blob}")
        return dataset_blob

    registry = load_dataset_registry(
        container_name="models",
        connection_string=connection_string
    )

    return registry["latest"]

def record_training_run(container_name, connection_string, model_version, dataset_version, metrics):
    """
    Save metadata about this training run so we can trace:
    dataset → models → metrics
    """

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container = blob_service_client.get_container_client(container_name)

    run_id = datetime.now(IST).strftime("v%Y%m%d_%H%M%S_%f")

    run_metadata = {
        "run_id": run_id,
        "timestamp": datetime.now(IST).isoformat(),
        "dataset_version": dataset_version,
        "model_version": model_version,
        "metrics": metrics
    }

    blob_path = f"training_runs/{run_id}.json"

    container.upload_blob(
        blob_path,
        json.dumps(run_metadata, indent=2),
        overwrite=True
    )

    print("Training run recorded:", blob_path)

def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map sentiment strings to numeric labels."""
    label_map = {
        "negative": 0,  # SELL
        "neutral": 1,   # HOLD
        "positive": 2   # BUY
    }
    df["label"] = df["sentiment"].map(label_map)
    if df["label"].isnull().any():
        raise ValueError("Found unmapped sentiment labels")
    return df

def split_dataset(df: pd.DataFrame):
    """Stratified train/validation split."""
    return train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

def build_tokenizer_and_model(model_name: str, num_labels: int = 3):
    """Load tokenizer and FinBERT models."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    return tokenizer, model

def tokenize_function(tokenizer, max_length: int):
    """Create a tokenization function for HF datasets."""
    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="longest",
            max_length=max_length
        )
    return _tokenize

def compute_metrics(eval_pred):
    """Evaluation metrics for sentiment classification."""
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }

def upload_model_to_blob(local_model_path, container_name, connection_string):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container = blob_service_client.get_container_client(container_name)

    # create a single version for the entire models upload
    model_version = datetime.now(IST).strftime("v%Y%m%d_%H%M%S")
    print(f"Uploading models version: {model_version}")

    try:
        container.create_container()
    except Exception:
        pass

    for root, dirs, files in os.walk(local_model_path):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, local_model_path)
            blob_path = f"finbert/{model_version}/{relative_path}"

            with open(file_path, "rb") as data:
                container.upload_blob(blob_path, data, overwrite=True)

    return model_version

def update_registry(container_name, connection_string, model_version):
    """Update models registry with latest staging models"""

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container = blob_service_client.get_container_client(container_name)

    registry_blob = "finbert/registry.json"

    try:
        blob_client = container.get_blob_client(registry_blob)
        data = blob_client.download_blob().readall()
        registry = json.loads(data)
    except Exception:
        registry = {"production": None, "staging": None}

    registry["staging"] = model_version

    blob_client = container.get_blob_client(registry_blob)
    blob_client.upload_blob(json.dumps(registry), overwrite=True)

    print("Registry updated. Staging models:", model_version)

def upload_metrics_to_blob(container_name, connection_string, model_version, metrics):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container = blob_service_client.get_container_client(container_name)

    blob_path = f"finbert/{model_version}/metrics.json"

    container.upload_blob(
        blob_path,
        json.dumps(metrics, indent=2),
        overwrite=True
    )

    print("Metrics uploaded:", blob_path)

def train():
    # Load config
    config_path = os.getenv("TRAIN_CONFIG", "config/training.yaml")
    config = load_config(config_path)
    print(
        type(config["learning_rate"]),
        type(config["batch_size"]),
        type(config["epochs"]),
        type(config["weight_decay"])
    )

    # Load and prepare dataset
    dataset_path = os.getenv("DATASET_PATH", "data/processed/processed_data.csv")

    if not os.path.exists(dataset_path):
        print("Dataset not found locally. Downloading from Azure Blob...")

        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

        dataset_blob = get_dataset_blob_path(connection_string)
        dataset_version = dataset_blob

        download_dataset_from_blob(
            container_name="models",
            blob_name=dataset_blob,
            local_path=dataset_path,
            connection_string=connection_string
        )
    else:
        # If dataset exists locally, we should still resolve the dataset version for tracking
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        dataset_blob = get_dataset_blob_path(connection_string)
        dataset_version = dataset_blob

    df = load_dataset(dataset_path)
    df = encode_labels(df)

    train_df, val_df = split_dataset(df)

    # Load models & tokenizer
    tokenizer, model = build_tokenizer_and_model(
        config["model_name"],
        num_labels=3
    )

    # Convert to HF Datasets
    train_ds = Dataset.from_pandas(train_df[["text", "label"]])
    val_ds = Dataset.from_pandas(val_df[["text", "label"]])

    tokenize = tokenize_function(tokenizer, config["max_length"])
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    cols = ["input_ids", "attention_mask", "label"]
    train_ds.set_format("torch", columns=cols)
    val_ds.set_format("torch", columns=cols)

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        evaluation_strategy="no",
        save_strategy="no",
        logging_steps=100,
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=2,
        num_train_epochs=config["epochs"],
        weight_decay=config["weight_decay"],
        logging_dir="logs",
        report_to="none",
        dataloader_num_workers=2
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    model_version = None
    metrics = None

    try:
        # mark training start
        upload_status({
            "status": "running",
            "version": None,
            "dataset_version": dataset_version
        })

        trainer.train()
        metrics = trainer.evaluate()

        print("Validation metrics:", metrics)

        # Save model locally
        os.makedirs(config["output_dir"], exist_ok=True)
        trainer.save_model(config["output_dir"])
        tokenizer.save_pretrained(config["output_dir"])

        print("Uploading model to Azure Blob Storage...")

        model_version = upload_model_to_blob(
            config["output_dir"],
            container_name="models",
            connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        )

        # update registry, metrics and lineage AFTER successful upload
        update_registry(
            container_name="models",
            connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
            model_version=model_version
        )

        upload_metrics_to_blob(
            container_name="models",
            connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
            model_version=model_version,
            metrics=metrics
        )

        record_training_run(
            container_name="models",
            connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
            model_version=model_version,
            dataset_version=dataset_version,
            metrics=metrics
        )

        # update status after successful training
        upload_status({
            "status": "completed",
            "version": model_version,
            "metrics": metrics
        })

    except Exception as e:
        upload_status({
            "status": "failed",
            "version": model_version,
            "error": str(e)
        })
        raise


    print("Model uploaded successfully.")


if __name__ == "__main__":
    train()