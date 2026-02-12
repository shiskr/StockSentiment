import argparse
from preprocessing.text_cleaner import clean_text
from preprocessing.save_file import save_csv
from training.train_sentiment import train
import pandas as pd

def run_training():
    processed_data = pd.DataFrame(clean_text())
    save_csv('./data/processed/processed_data.csv', processed_data)

    data = pd.read_csv('./data/processed/processed_data.csv')

    label_map = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }

    data["label"] = data["sentiment"].map(label_map)
    data.to_csv('./data/processed/processed_data_labeled.csv', index=False)

    train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train"])
    args = parser.parse_args()

    if args.command == "train":
        run_training()