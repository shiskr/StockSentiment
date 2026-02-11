import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "model/finbert"

_tokenizer = None
_model = None

LABELS = {0: "SELL", 1: "HOLD", 2: "BUY"}

def load_model():
    global _tokenizer, _model

    if _model is None or _tokenizer is None:
        print("Loading model once...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        _model.eval()


def predict(text: str):
    load_model()
    inputs = _tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = _model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    label = LABELS[probs.argmax().item()]
    return label, probs.tolist()