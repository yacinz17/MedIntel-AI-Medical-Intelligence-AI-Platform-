# MedIntel AI

MedIntel AI is a portfolio-oriented medical AI prototype that demonstrates three separate workflows:

- **Structured-data ML** — a Random Forest pipeline for a Pima-style diabetes dataset.
- **Chest X-ray classification** — an optional EfficientNetB0 training/inference pipeline.
- **Medical text extraction** — a lightweight rule-based prototype for symptoms, dosage mentions, and medication phrases.

> **Safety:** This repository is for research, learning and portfolio demonstration only. It is **not a medical device**, does **not provide a diagnosis**, and must not be used for clinical decisions.

## Why this version exists

The original notebook mixed incompatible dependency versions and contained a FastAPI endpoint that returned a placeholder prediction. This refactor separates experimentation, training, inference, API and UI so that the repository only claims capabilities that are actually implemented.

## Architecture

```text
Streamlit UI
    |
    v
FastAPI API
    |-------------------|-------------------|
    v                   v                   v
Diabetes model      X-ray model       Text extraction
(joblib)            (Keras, optional)  (rule-based)
```

## Project structure

```text
app/
  main.py             # FastAPI application
  schemas.py          # Request models
  services.py         # Inference and text-analysis logic
  settings.py         # Paths and safety disclaimer
  streamlit_app.py    # Streamlit UI
scripts/
  train_diabetes.py   # Train structured-data model
  train_xray.py       # Train EfficientNet model
tests/
  test_api.py
models/               # Local model artefacts; not committed
data/                 # Local datasets; not committed
Medical_AI_.ipynb     # Original experimental notebook
```

## Quick start

### 1. Create an environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Start the API

```bash
uvicorn app.main:app --reload
```

Open API docs at `http://127.0.0.1:8000/docs`.

### 3. Start the UI

In a second terminal:

```bash
streamlit run app/streamlit_app.py
```

## Train the structured-data model

Place a compatible CSV at `data/diabetes.csv` with these columns:

```text
Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome
```

Then run:

```bash
python scripts/train_diabetes.py
```

The model will be saved as `models/diabetes_rf.joblib`. Model files are deliberately excluded from Git because training data provenance and model evaluation should remain explicit.

## Train the X-ray model

Install the optional image dependency:

```bash
pip install -r requirements-image.txt
```

Expected layout:

```text
data/chest_xray/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
```

Then run:

```bash
python scripts/train_xray.py
```

The resulting model is stored at `models/chest_xray_effnet.keras`.

## API endpoints

- `GET /health` — service health and safety disclaimer.
- `GET /models` — reports whether local model artefacts are available.
- `POST /predict/diabetes` — structured-data inference when the model exists.
- `POST /predict/xray` — JPEG/PNG X-ray inference when the optional model exists.
- `POST /analyse/text` — lightweight text extraction.

If a required model file is missing, the API returns **503** with an explicit training instruction instead of fabricating a prediction.

## Tests

```bash
pip install -e .[dev]
pytest
```

## Docker

API only:

```bash
docker build -t medintel-ai .
docker run --rm -p 8000:8000 medintel-ai
```

## Current limitations

- No model in this repository is clinically validated.
- The original notebook does not contain sufficient evidence to claim production-level diagnostic performance.
- X-ray inference requires a separately trained model artefact.
- Text analysis is a demonstration extractor, not clinical NLP.
- Before any healthcare deployment, data governance, bias, calibration, external validation, monitoring, security, and relevant medical-device regulation would need to be addressed.

## Portfolio value

This refactor demonstrates:

- separation of notebook experimentation from application code;
- reproducible training scripts;
- FastAPI service design and request validation;
- Streamlit UI integration;
- model lifecycle handling;
- graceful failure when artefacts are unavailable;
- automated API tests;
- containerisation basics;
- explicit safety and scope boundaries for medical AI.
