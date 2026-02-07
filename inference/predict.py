from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "model/finbert"

_tokenizer = None
_model = None

def load_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        _model.eval()

def predict(text):
    load_model()

    inputs = _tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = _model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    id2label = {0: "SELL", 1: "HOLD", 2: "BUY"}
    return id2label[probs.argmax().item()], probs.tolist()