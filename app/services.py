from __future__ import annotations

import io
import re
from functools import lru_cache

import joblib
import numpy as np
from PIL import Image

from app.settings import DIABETES_MODEL_PATH, XRAY_MODEL_PATH

DIABETES_FEATURE_ORDER = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


@lru_cache(maxsize=1)
def load_diabetes_model():
    if not DIABETES_MODEL_PATH.exists():
        return None
    return joblib.load(DIABETES_MODEL_PATH)


@lru_cache(maxsize=1)
def load_xray_model():
    if not XRAY_MODEL_PATH.exists():
        return None
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed. Install requirements-image.txt to use X-ray inference."
        ) from exc
    return tf.keras.models.load_model(XRAY_MODEL_PATH)


def predict_diabetes(features: dict[str, float]) -> dict:
    model = load_diabetes_model()
    if model is None:
        raise FileNotFoundError(
            "Diabetes model not found. Train it with scripts/train_diabetes.py first."
        )
    values = np.array([[features[k] for k in DIABETES_FEATURE_ORDER]], dtype=float)
    probability = float(model.predict_proba(values)[0, 1])
    prediction = int(probability >= 0.5)
    return {"prediction": prediction, "probability": round(probability, 4)}


def predict_xray(image_bytes: bytes) -> dict:
    model = load_xray_model()
    if model is None:
        raise FileNotFoundError(
            "Chest X-ray model not found. Train it with scripts/train_xray.py first."
        )
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    x = np.asarray(image, dtype=np.float32)
    # EfficientNet includes its own rescaling in modern tf.keras applications.
    x = np.expand_dims(x, axis=0)
    probability = float(np.ravel(model.predict(x, verbose=0))[0])
    label = "PNEUMONIA" if probability >= 0.5 else "NORMAL"
    confidence = probability if label == "PNEUMONIA" else 1.0 - probability
    return {
        "label": label,
        "pneumonia_probability": round(probability, 4),
        "confidence": round(float(confidence), 4),
    }


SYMPTOM_TERMS = [
    "chest pain", "shortness of breath", "cough", "fever", "fatigue",
    "headache", "nausea", "vomiting", "dizziness", "pain"
]


def analyse_text(text: str) -> dict:
    lowered = text.lower()
    symptoms = sorted({term for term in SYMPTOM_TERMS if term in lowered})
    dosages = re.findall(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml)\b", text, flags=re.I)
    medication_phrases = re.findall(
        r"(?:prescribed|taking|take|on)\s+([A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z][A-Za-z0-9-]*){0,2})",
        text,
        flags=re.I,
    )
    return {
        "symptoms_detected": symptoms,
        "dosages_detected": dosages,
        "medication_phrases": medication_phrases[:10],
        "character_count": len(text),
        "method": "rule-based prototype",
    }
