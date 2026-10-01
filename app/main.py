from fastapi import FastAPI, File, HTTPException, UploadFile

from app.schemas import DiabetesFeatures, TextRequest
from app.services import analyse_text, predict_diabetes, predict_xray
from app.settings import DIABETES_MODEL_PATH, DISCLAIMER, XRAY_MODEL_PATH

app = FastAPI(
    title="MedIntel AI API",
    version="0.1.0",
    description=DISCLAIMER,
)


@app.get("/health")
def health():
    return {"status": "ok", "disclaimer": DISCLAIMER}


@app.get("/models")
def model_status():
    return {
        "diabetes_model_ready": DIABETES_MODEL_PATH.exists(),
        "xray_model_ready": XRAY_MODEL_PATH.exists(),
    }


@app.post("/predict/diabetes")
def diabetes(payload: DiabetesFeatures):
    mapping = {
        "Pregnancies": payload.pregnancies,
        "Glucose": payload.glucose,
        "BloodPressure": payload.blood_pressure,
        "SkinThickness": payload.skin_thickness,
        "Insulin": payload.insulin,
        "BMI": payload.bmi,
        "DiabetesPedigreeFunction": payload.diabetes_pedigree_function,
        "Age": payload.age,
    }
    try:
        result = predict_diabetes(mapping)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {**result, "disclaimer": DISCLAIMER}


@app.post("/predict/xray")
async def xray(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG or PNG image.")
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit.")
    try:
        result = predict_xray(data)
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process image: {exc}") from exc
    return {**result, "disclaimer": DISCLAIMER}


@app.post("/analyse/text")
def text_analysis(payload: TextRequest):
    return {**analyse_text(payload.text), "disclaimer": DISCLAIMER}
