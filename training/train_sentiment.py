import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
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
    """Load tokenizer and FinBERT model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        use_safetensors=True
    )
    return tokenizer, model


def tokenize_function(tokenizer, max_length: int):
    """Create a tokenization function for HF datasets."""
    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
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


def train():
    # Load config
    config = load_config("config/training.yaml")
    print(
        type(config["learning_rate"]),
        type(config["batch_size"]),
        type(config["epochs"]),
        type(config["weight_decay"])
    )

    # Load and prepare dataset
    df = load_dataset("data/processed/processed_data.csv")
    df = encode_labels(df)

    train_df, val_df = split_dataset(df)

    # Load model & tokenizer
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

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        # evaluation_strategy="epoch",
        # save_strategy="epoch",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        weight_decay=config["weight_decay"],
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        logging_dir="logs",
        report_to="none"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()

    # Save model
    os.makedirs(config["output_dir"], exist_ok=True)
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    print(f"Training complete. Model saved to {config['output_dir']}")


if __name__ == "__main__":
    train()
