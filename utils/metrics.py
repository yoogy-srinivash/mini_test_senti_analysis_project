def apply_confidence_threshold(label, confidence, threshold):
    if confidence < threshold:
        return "UNCERTAIN"
    return label
