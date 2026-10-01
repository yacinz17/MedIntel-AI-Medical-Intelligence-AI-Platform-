from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_models_endpoint():
    response = client.get("/models")
    assert response.status_code == 200
    body = response.json()
    assert "diabetes_model_ready" in body
    assert "xray_model_ready" in body


def test_text_analysis():
    response = client.post(
        "/analyse/text",
        json={"text": "Patient reports chest pain. Prescribed aspirin 81 mg daily."},
    )
    assert response.status_code == 200
    body = response.json()
    assert "chest pain" in body["symptoms_detected"]
    assert "81 mg" in body["dosages_detected"]
