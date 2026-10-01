from pathlib import Path
import json

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

DATA_DIR = Path("data/chest_xray/test")
MODEL_PATH = Path("models/chest_xray_effnet.keras")
RESULTS_DIR = Path("results")


def main():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow is required. Install it with: pip install -r requirements-image.txt"
        ) from exc

    if not DATA_DIR.exists():
        raise SystemExit(f"Test data not found: {DATA_DIR}")

    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test dataset...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        image_size=(224, 224),
        batch_size=16,
        label_mode="binary",
        shuffle=False,
    )

    class_names = test_ds.class_names
    print(f"Classes: {class_names}")

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Running predictions on the full test set...")
    y_prob = np.ravel(model.predict(test_ds, verbose=1))
    y_true = np.concatenate([np.ravel(labels.numpy()) for _, labels in test_ds]).astype(int)
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== Test-set evaluation ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print("\nConfusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    metrics = {
        "dataset": "Chest X-Ray Images (Pneumonia) test split",
        "classes": class_names,
        "threshold": 0.5,
        "n_samples": int(len(y_true)),
        "accuracy": round(float(accuracy), 4),
        "precision_pneumonia": round(float(precision), 4),
        "recall_pneumonia": round(float(recall), 4),
        "f1_pneumonia": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": cm.tolist(),
    }

    out_path = RESULTS_DIR / "xray_test_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved metrics to {out_path}")


if __name__ == "__main__":
    main()
