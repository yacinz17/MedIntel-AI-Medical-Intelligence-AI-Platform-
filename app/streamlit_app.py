import requests
import streamlit as st

API_URL = st.sidebar.text_input("API URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="MedIntel AI", page_icon="🩺", layout="wide")
st.title("MedIntel AI")
st.caption("Research and portfolio prototype — not for clinical use or diagnosis.")

try:
    status = requests.get(f"{API_URL}/models", timeout=3).json()
    st.sidebar.success("API connected")
    st.sidebar.json(status)
except Exception:
    st.sidebar.error("API unavailable. Start FastAPI with: uvicorn app.main:app --reload")

tab1, tab2, tab3 = st.tabs(["Structured data", "Chest X-ray", "Medical text"] )

with tab1:
    st.subheader("Diabetes risk model")
    c1, c2 = st.columns(2)
    with c1:
        pregnancies = st.number_input("Pregnancies", min_value=0.0, value=1.0)
        glucose = st.number_input("Glucose", min_value=0.0, value=120.0)
        blood_pressure = st.number_input("Blood pressure", min_value=0.0, value=70.0)
        skin_thickness = st.number_input("Skin thickness", min_value=0.0, value=20.0)
    with c2:
        insulin = st.number_input("Insulin", min_value=0.0, value=80.0)
        bmi = st.number_input("BMI", min_value=0.0, value=25.0)
        dpf = st.number_input("Diabetes pedigree function", min_value=0.0, value=0.5)
        age = st.number_input("Age", min_value=0.0, value=30.0)
    if st.button("Run structured-data model"):
        payload = {
            "pregnancies": pregnancies, "glucose": glucose,
            "blood_pressure": blood_pressure, "skin_thickness": skin_thickness,
            "insulin": insulin, "bmi": bmi,
            "diabetes_pedigree_function": dpf, "age": age,
        }
        try:
            r = requests.post(f"{API_URL}/predict/diabetes", json=payload, timeout=20)
            st.json(r.json())
        except Exception as exc:
            st.error(str(exc))

with tab2:
    st.subheader("Chest X-ray classifier")
    image = st.file_uploader("Upload JPEG or PNG", type=["jpg", "jpeg", "png"])
    if image and st.button("Run X-ray model"):
        try:
            r = requests.post(
                f"{API_URL}/predict/xray",
                files={"file": (image.name, image.getvalue(), image.type)},
                timeout=60,
            )
            st.json(r.json())
        except Exception as exc:
            st.error(str(exc))

with tab3:
    st.subheader("Medical text extraction")
    text = st.text_area(
        "Text",
        "Patient complains of chest pain and shortness of breath. Prescribed aspirin 81 mg daily.",
        height=160,
    )
    if st.button("Analyse text"):
        try:
            r = requests.post(f"{API_URL}/analyse/text", json={"text": text}, timeout=20)
            st.json(r.json())
        except Exception as exc:
            st.error(str(exc))
