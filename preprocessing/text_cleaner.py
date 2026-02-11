import os

def clean_text():
    data = []

    DATA_FOLDER = "./data/raw/FinancialPhraseBank-v1.0"

    for filename in os.listdir(DATA_FOLDER):
        if not filename.startswith("Sentences"):
            continue

        file_path = os.path.join(DATA_FOLDER, filename)

        with open(file_path, "r", encoding="latin1") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                text, label = line.rsplit("@", 1)
                data.append({
                    "text": text.strip(),
                    "sentiment": label.strip()
                })
    return data