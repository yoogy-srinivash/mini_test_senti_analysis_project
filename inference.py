import torch
import torch.nn.functional as F

from model_loader import model, tokenizer
from config import MAX_LENGTH, CONFIDENCE_THRESHOLD
from utils.metrics import apply_confidence_threshold


def tokenize(text):
    return tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )


def predict_sentiment(text):
    inputs = tokenize(text)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    logits = outputs.logits
    probs = F.softmax(logits, dim=1)

    confidence, pred = torch.max(probs, dim=1)
    label = model.config.id2label[pred.item()]

    final_label = apply_confidence_threshold(
        label,
        confidence.item(),
        CONFIDENCE_THRESHOLD
    )

    return {
        "label": final_label,
        "confidence": round(confidence.item(), 3)
    }