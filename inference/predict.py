from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

_model = None
_tokenizer = None


def load_model():
    global _model, _tokenizer

    if _model is None:
        print("Loading model once...")
        model_path = "model/finbert"

        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        _model = AutoModelForSequenceClassification.from_pretrained(model_path)
        _model.eval()


def predict(text):
    load_model()

    inputs = _tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = _model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs).item()

    mapping = {0: "SELL", 1: "HOLD", 2: "BUY"}

    return mapping[pred]